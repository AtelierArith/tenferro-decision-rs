# jeff-infer Design

**Status:** design proposal (implements the package's Jeff requirements)  
**Date:** 2026-10-04  
**Sources:** `09_JEFFCLIENT_ANALYSIS.md`, `12_TENFERRO_GATED_DELTA.md`, `11_TENFERRO_API_SURVEY.md`, `02_SPECIFICATION.md` §5

`jeff-infer` is the Jeff typed-decision engine: a text-only Qwen3.5 hybrid
(full-attention and Gated DeltaNet layers) with Jeff's trained readout. In the
package's sequence it follows Laya; it is the first consumer of
`tenferro-gated-delta`.

---

## 1. Scope and responsibilities

Owns:

- checkpoint loading (`config.json`, `decision_config.json`,
  `model.safetensors`, `readout.safetensors`)
- the supported Qwen3.5 subset: embedding, full-attention layers,
  Gated DeltaNet layers, RMSNorm, partial RoPE, SiLU MLP, final norm
- Jeff readout and decision calibration
- prepared-token inference (`input_ids` + `attention_mask`)
- Laya-independent execution plans and workspaces
- later, optional tokenizer support

Does not own:

- DeltaNet execution (in `tenferro-gated-delta`)
- shared primitives (in `tenferro-infer`)
- generation, KV cache, images, training (package non-goals)

---

## 2. Crate layout and public API

```text
crates/jeff-infer/
├── src/
│   ├── lib.rs
│   ├── config.rs        # text_config + decision_config
│   ├── checkpoint.rs    # tensor naming, strict shape validation
│   ├── model.rs         # JeffModel (weights + plan)
│   ├── context.rs       # JeffContext (model + workspace)
│   ├── layers.rs        # full-attention and delta layers
│   ├── attention.rs     # full attention (partial RoPE, GQA, Q gate)
│   ├── readout.rs       # readout + calibration + answers
│   ├── plan.rs          # shape buckets, prepared plans
│   └── tokenizer.rs     # optional, later
└── tests/
```

Public surface (proposed):

```rust
pub struct JeffModel { /* weights: Arc<JeffWeights>, plan: Arc<JeffPlan> */ }
pub struct JeffContext { /* model: Arc<JeffModel>, workspace: JeffWorkspace */ }

impl JeffModel {
    pub fn load(path: &Path, backend: &mut dyn BackendSession) -> Result<Self>;
}

impl JeffContext {
    /// Prepared-token entry point.
    pub fn logits(&mut self, input_ids: &Tensor, mask: &Tensor) -> Result<Tensor>;
    pub fn decide(&mut self, input_ids: &Tensor, mask: &Tensor, qs: &[Question]) -> Result<Vec<Answer>>;
}
```

The `DecisionEngine` impl (per `10_DECISION_ABSTRACTION.md`) wraps
`JeffContext`. Initially it accepts prepared tokens; the `State` decision for
natural-language input is left to the trait design.

---

## 3. Config and checkpoint

`config.json`:

- `model_type == "qwen3_5"`
- `text_config`: `hidden_size`, `head_dim`, `num_attention_heads`,
  `num_key_value_heads`, `linear_num_key_heads`, `linear_num_value_heads`,
  `linear_key_head_dim`, `linear_value_head_dim`, `rms_norm_eps`,
  `rope_parameters.rope_theta`, `layer_types`
- `attention_bias == false`, `hidden_act == "silu"`,
  `rope_parameters.rope_type == "default"`

`decision_config.json`:

- `format_version == 1`
- `temperature` finite and positive
- `max_options` within `1 .. min(255, readout columns)`

Validations also enforce `value_heads % key_heads == 0` and
`heads % kv_heads == 0`. Derived: `rotary_dim = head_dim * partial_rotary_factor`.

Checkpoint tensors use the `language_model.` prefix and the naming in
`09_JEFFCLIENT_ANALYSIS.md` §3; the loader validates names and shapes strictly
and loads `readout.safetensors["weight"]`.

Weights are stored `(in, out)`; a load-time transpose/pack chooses the canonical
runtime layout.

---

## 4. Forward pipeline

```text
input_ids, attention_mask  (B, L), left-padded
for each row:
    first  = first active position (leading zeros; trim policy)
    hidden = embedding[:, ids[first..]] + 1     # (hidden, L')
    hidden = encoder(hidden, mask)
    hidden = final RMSNorm(hidden[:, last])
    scores = readout^T hidden                   # (options,)
```

Encoder layer stack, ordered by `layer_types`:

```text
for layer:
    normalized = RMSNorm_centered(x, input_norm)
    mixed = full_attention(normalized) or gated_delta(normalized)
    residual = x + mixed
    normalized = RMSNorm_centered(residual, post_norm)
    residual = residual + MLP(normalized)
    x = residual
```

The last encoder output is normalized and only its last position is read out
(the reference optimization). The package requires this to be validated as a
semantic shortcut (`09_JEFFCLIENT_ANALYSIS.md` §12).

`RMSNorm` has two modes: **centered** (`1 + weight`) for input/post/final and
Q/K norms, and **non-centered** (`weight`) inside Gated DeltaNet. The
difference is load-bearing.

---

## 5. Full-attention layer

```text
qgate = q_proj^T x reshaped to (2*head_dim, heads, L)   # first half query, second gate
q = RoPE(RMSNorm_centered(qgate[0:head_dim], q_norm))
k = RoPE(RMSNorm_centered(k_proj^T x, k_norm))
v = v_proj^T x
scores mask: key j <= query i and mask[j] == 1
for head:
    scores = (k_gqa^T q) / sqrt(head_dim) + mask
    probs  = softmax over keys
    out    = v_gqa @ probs * sigmoid(qgate_gate)
out = o_proj^T out
```

- **Q gate**: `q_proj` emits twice the head width; the second half is a sigmoid
  gate on the value product.
- **GQA**: query head `h` maps to KV head `ceil(h / (heads / kv_heads))`
  (consecutive grouping).
- **Partial RoPE**: only the first `rotary_dim` channels; pairs
  `(i, i + rotary_dim/2)`; positions are absolute over the trimmed row.
- Masks are compact metadata in the optimized path, not a dense `L × L` matrix.

The `tenferro-infer` RoPE variant and attention primitive are used here.

---

## 6. Gated DeltaNet layer

Fully delegated to `tenferro-gated-delta` (`12_TENFERRO_GATED_DELTA.md`): causal
depthwise convolution + SiLU, Q/K L2 normalization, `beta`/`decay`
precomputation, and the Delta core (reference, chunked, recurrent, CUDA). This
crate supplies normalized input, mask, prepared weights, and receives the layer
output.

The mask is applied by zeroing invalid positions before the Delta projection,
matching the reference.

---

## 7. MLP

```text
down_proj^T ( SiLU(gate_proj^T x) * (up_proj^T x) )
```

`SiLU` is a composed primitive until profiling justifies a fused gate.

---

## 8. Readout and decisions

`logits` returns raw `(B, max_options)` scores; `decide` applies per question:

```text
p = softmax((logits - max) / temperature)   # active options only
```

- **choice**: `argmax`, confidence follows the Jeff formula
  `clamp((p_best - 1/n) / (1 - 1/n), 0, 1)`
- **noul**: `noul = p[true]` (columns false, true)
- **score**: zero-based expected value `sum(level * p)` with a confidence

These are the `decision-core` answer types from `10_DECISION_ABSTRACTION.md`.
Option order is caller-defined and must match checkpoint column order.

---

## 9. Prepared-token API and tokenizer

The initial engine accepts prepared `input_ids` and `attention_mask` only, as the
reference native path does. General text tokenization is out of the initial
scope (`02_SPECIFICATION.md` §5.1). If added later, it is a separate module and
must not become a dependency of the prepared-token path.

---

## 10. CPU optimization plan

- load-time weight packing; one canonical runtime layout
- prepared GEMM for projections; batched/grouped products for heads via
  `dot_general` batch dims
- workspace classes: hidden/residual, projection scratch, attention scratch,
  Delta state/conv scratch, MLP intermediate
- fused residual + RMSNorm; composed softmax; fused RoPE/gate where profiling
  justifies it
- shape-bucketed plans for common lengths; generic fallback for correctness
- thread policy that avoids provider oversubscription

Reference first (`06_ROADMAP.md` Phase 5), then `tenferro-gated-delta`
optimization (Phase 6).

---

## 11. CUDA plan

- device-resident weights, RoPE tables, workspaces
- CUDA full attention and partial RoPE/grouped-KV preparation
- CUDA causal convolution and specialized Gated Delta kernel
  (`12_TENFERRO_GATED_DELTA.md` §10)
- device-resident readout; compact download
- no hidden CPU fallback, no intermediate host transfer, explicit
  synchronization boundaries

---

## 12. Workspace and prepared plans

Same Model/Context pattern as Laya. Plans are shape-specialized; GPU workspaces
are stream-owned with asynchronous lifetime rules. `RMSNorm`, RoPE tables and
`a_decay = -exp(A_log)` are prepared at load time.

---

## 13. Testing and parity

Reference: JeffClient.jl CPU/Metal or an independent PyTorch reference
(`05_TESTING_BENCHMARKS.md` §2).

- intermediate activations: embedding, RMSNorm, projections, RoPE, full
  attention, Delta pre-conv, convolution, delta recurrence/chunk result, MLP,
  final norm, readout logits
- Delta cross-formulation parity via `tenferro-gated-delta` tests
- shape regression: `L = 1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 512`,
  batch 1 and > 1, left padding, interior mask holes
- benchmark matrix `(batch, seq) = (1,128), (1,256), (1,512), (4,256)`, with
  padded full-length and trimmed variants reported separately
- semantic decision regression across choice/noul/score

---

## 14. decision-core integration

`JeffContext` implements the shared `DecisionEngine`. Its input follows the
`State` decision in `10_DECISION_ABSTRACTION.md` §4: prepared tokens in the
initial engine, natural-language state only if a tokenizer is added.

---

## 15. Open questions

- [ ] `layer_types` pattern in the target checkpoint and whether any layer
      combination beyond full/linear must be supported.
- [ ] Whether the engine ever needs natural-language tokenization, and if so
      which Qwen tokenizer assets.
- [ ] Tolerance policy for CPU and CUDA, and against which reference.
- [ ] Which fusions to move into extension ops (attention, residual+RMSNorm,
      MLP gate) versus composed tenferro ops.
- [ ] How `decision-core`'s `State` should represent prepared tokens versus text.
- [ ] Whether Apple GPU support follows after CUDA or is deferred further.
