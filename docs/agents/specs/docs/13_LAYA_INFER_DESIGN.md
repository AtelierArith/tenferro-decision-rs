# laya-infer Design

**Status:** design proposal (implements the package's Laya requirements)  
**Date:** 2026-10-04  
**Source reference:** `extern/Laya.jl` at commit `e9af31a4ccfe8692c2dbe8e7ddc456776f54df05`  
**Related:** `02_SPECIFICATION.md` §4, `04_MODEL_MAPPING.md` §1, `10_DECISION_ABSTRACTION.md`, `11_TENFERRO_API_SURVEY.md`

`laya-infer` is the Laya typed-decision engine: a modern-BERT encoder plus decision
head, scorer and action head, with a native tokenizer and prompt builder. It is
the first production milestone in the package and must run CPU inference without
Python or Julia, then CUDA.

---

## 1. Scope and responsibilities

Owns:

- checkpoint loading (`model.safetensors`, `encoder/config.json`,
  `rl_agent_config.json`, `mlx_config.json`, tokenizer assets)
- tokenizer (pure Rust)
- prompt construction and Python-compatible serialization
- embedding and embedding normalization
- ModernBERT encoder layers (full and sliding attention, non-traditional RoPE)
- typed embedding and decision-head transformer layers
- marker gather, scorer, action head
- calibration and typed answers
- Laya-specific execution plans and workspaces

Does not own:

- shared inference primitives (in `tenferro-infer`)
- backend/runtime selection (the backend is injected)
- Jev transport (in `jev-client`)

---

## 2. Crate layout and public API

```text
crates/laya-infer/
├── src/
│   ├── lib.rs
│   ├── config.rs        # EncoderConfig, AgentConfig, calibration
│   ├── checkpoint.rs    # WeightReader, sanitize, strict shape validation
│   ├── tokenizer.rs     # byte-level / Metaspace BPE
│   ├── prompt.rs        # py_json, build_prefix/build_sequence, markers
│   ├── collate.rs       # batch assembly
│   ├── model.rs         # LayaModel (weights + plan)
│   ├── context.rs       # LayaContext (model + workspace)
│   ├── layers.rs        # encoder / head layers
│   ├── heads.rs         # scorer, action head, calibration, answers
│   ├── plan.rs          # shape buckets, prepared plans
│   └── extension.rs     # optional fused extension ops
└── tests/
```

Public surface (proposed):

```rust
pub struct LayaModel { /* weights: Arc<LayaWeights>, plan: Arc<LayaPlan> */ }
pub struct LayaContext { /* model: Arc<LayaModel>, workspace: LayaWorkspace */ }

pub struct Agent {
    pub model: LayaModel,
    pub tokenizer: Tokenizer,
    pub calibration: Calibration,
    pub config: AgentConfig,
}

impl Agent {
    pub fn load(path: &Path, backend: &mut dyn BackendSession) -> Result<Self>;

    /// Prepare questions into a padded batch (ids, masks, marker positions, qtype).
    pub fn prepare(&self, state: &State, qs: &[Question]) -> Result<Vec<PreparedItem>>;

    /// Batch forward: raw marker logits and action probabilities.
    pub fn forward(&mut self, batch: &Batch) -> Result<(Tensor, Tensor)>;

    /// Full decision entry point, mirroring upstream `system_one`.
    pub fn system_one(&mut self, state: &State, qs: &[Question]) -> Result<Vec<Answer>>;
}
```

`LayaContext` owns the mutable workspace; `Agent` is the convenience type that
also owns the tokenizer and calibration. The `DecisionEngine` impl (per
`10_DECISION_ABSTRACTION.md`) wraps `Agent`.

---

## 3. Checkpoint, config and strict loading

Required files (rejected as incomplete otherwise):

```text
model.safetensors
rl_agent_config.json
encoder/config.json
mlx_config.json          (optional metadata)
tokenizer/               (tokenizer assets)
```

`rl_agent_config.json` supplies `head_layers` (default 2), `act_costs` (action
count), `max_len` (512), `head_max_len` (192) and the calibration temperatures.
Validation (`load`):

- `head_layers` present and `encoder` present
- `4 < head_max_len < max_len <= max_position_embeddings`
- the three base temperatures are finite and positive
- calibration values are clamped to `[0.5, 5.0]` with a warning (shipped values
  below 1 would otherwise distort confidence)

`encoder/config.json` (`EncoderConfig`) validation:

- `model_type == "modernbert"`
- `hidden_activation == "gelu"`
- `hidden_size % num_attention_heads == 0` and the head dimension is even
- `layer_types` length equals `num_hidden_layers` and uses only
  `full_attention` / `sliding_attention`
- RoPE `rope_type == "default"` for every used kind

The Rust loader mirrors `WeightReader`: every checkpoint tensor must be consumed
exactly once with the expected (reversed) shape, and leftover tensors are an
error. `in_proj_weight` / `in_proj_bias` and the scorer/action-head prefixes are
normalized (`sanitize_weights`).

All weights are stored `(in, out)` (PyTorch/MLX `(out, in)` reversed), as in
`09_JEFFCLIENT_ANALYSIS.md` §3 for Jeff; the same convention applies to Laya.

---

## 4. Tokenizer

The Rust tokenizer must reproduce the Julia/upstream token IDs bit-exactly,
including:

- byte-level BPE and Metaspace BPE variants
- `[CLS]`, `[SEP]`, `[MASK]` special tokens and IDs
- padding token ID
- truncation and the mask-token replacement used by the prompt builder

Golden tests compare token IDs, padding, truncation, special tokens and marker
positions against the reference (`02_SPECIFICATION.md` §4.2).

---

## 5. Prompt construction and serialization

Exactness matters: upstream compares `json.dumps`-style strings, so the Rust
implementation must reproduce Python float formatting and escaping.

- `serialize_state`: a string passes through; any other value is serialized.
- `render_criterion`: string passthrough or Python-compatible JSON.
- `render_options`:
  - `choice`: `"key: description"` when a description is present, else `"key"`
  - `score`: `"level <i-1>: description"`
  - `noul`: `"false: ..."` / `"true: ..."` with the default texts
- `build_prefix`:
  `[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP]`
  with per-option truncation (`head_max_len`, default 192) and a fallback that
  reserves at least 16 tokens
- `build_sequence`:
  `[CLS] <type> ... [SEP] ... [SEP] state [SEP]`, truncated to `max_len`
  (default 512); marker positions are 0-based

A Rust `py_json`-equivalent formatter is required (shortest round-trip float
digits, scientific thresholds, `\uXXXX` escaping with surrogates for non-ASCII
when `ascii=true`, Python's `", "` / `": "` separators).

---

## 6. Collation and batch layout

Tensors use column-major `(d, L, B)` (Python `(B, L, d)` with axes reversed):

- `input_ids` `(L, B)`, `attention_mask` `(L, B)` (bool)
- `marker_pos` `(K, B)`, `marker_mask` `(K, B)`, with `K >= 2`
- `qtype` `(B,)` (`choice=0`, `score=1`, `noul=2`)

Collation pads to the maximum length and marker count in the batch with the
checkpoint pad ID and `false` mask.

---

## 7. Model forward

### 7.1 Encoder

```text
x = embed_norm(embed(tok_embeddings, input_ids))     # (d, L, B)
h = first_layer.attn_norm(x)  or x                    # first layer has no attn_norm
for i, layer in layers:
    next_norm = layers[i+1].attn_norm or final_norm
    y, h = layer(x, h, mask_i, next_norm)
    x = y
return h                                              # final_norm(encoder output)
```

Each layer (pre-norm):

```text
qkv = Wqkv(h)                                        # (3d, L, B)
a   = attention(qkv, heads, rope_base, mask, hd^-0.5)
z   = x + Wo(a)
hn  = mlp_norm(z)
u   = Wi(hn)                                         # (2*I, L, B)
g   = gelu(first half of u) * second half of u       # GeGLU
y   = z + Wo_mlp(g)
return y, next_norm(y)
```

Key details:

- **Fused residual + next normalization**: the residual sum and the next
  layer's norm are one logical step and a CPU/GPU fusion target.
- **First layer** has `attn_norm == nothing`.
- **RoPE is non-traditional**: within each head, the pair `(i, i + head_dim/2)`
  is rotated by `position * base^(-2i/head_dim)`; base is `global_rope_theta`
  (160000) for full attention and `local_rope_theta` (10000) for sliding, with
  per-kind `rope_parameters` overrides. Positions are `0 .. L-1`.
- **Attention masks**:
  - full: every valid key for every query (`valid_k`, broadcast over queries)
  - sliding: `|i - j| <= local_attention / 2` for valid queries, all valid keys
    for padded queries
  - masks are built as `(L_k, L_q, B)` bool; the optimized path stores compact
    validity + window metadata and skips out-of-window tiles
    (`02_SPECIFICATION.md` §4.5)
- **GeGLU**: `Wi` produces `2I`; `gelu(first half) * second half` (note the
  order), followed by `Wo_mlp` `I -> d`.
- GELU is the **erf-based exact** GELU (`x * (1 + erf(x/√2)) / 2`), matching
  the MLX kernel; the Rust primitive must use the same evaluation.

### 7.2 Typed head

```text
te = type_emb[:, qtype]                              # (d, B)
h  = encoder_output + te
for each head layer:
    h = head_layer(h, mask)                          # attention without RoPE
```

Head layer internals mirror a PyTorch `TransformerEncoderLayer` (pre-norm):

```text
n = norm1(h)
a = out_proj(attention(in_proj(n), heads, no_rope, mask, hd^-0.5))
z = h + a
hn = norm2(z)
y = z + linear2(relu(linear1(hn)))                   # NOTE: ReLU, not GELU
```

The head uses **ReLU**; the encoder and scorer use GELU. Head count is
`max(1, d / 64)`.

### 7.3 Marker gather, scorer, action head

```text
markers = h[:, marker_pos, :]                        # (d, K, B), 0-based positions
s0 = scorer_norm(markers)
s1 = scorer1(s0)
g1 = gelu(s1)
s2 = scorer2(g1)                                     # (1, K, B)
logits = s2[0, :, :]                                  # (K, B)
logits = where(marker_mask, logits, -1e4)
p = softmax(logits over K)
```

Pooled features for the action head:

```text
h1      = h[:, 0, :]                                  # first token (d, B)
top1    = p[0]; top2 = p[1] (partial sort desc, K >= 2)
entropy = -sum(p * log(max(p,1e-9))) / log(max(sum(marker_mask), 2))
features = [top1; top1 - top2; entropy; k/255]        # (4, B)
pooled  = [h1; features]                              # (d+4, B)
action  = act2(gelu(act1(pooled)))                    # (n_actions, B)
```

`K` is padded to at least 2 for single-option choices. Used markers are those
with `marker_mask == true`.

---

## 8. Calibration and answers

For each question with `k` used options and `qtype`:

```text
scale = temperature_by_options[bucket(qtype, k)] or temperature[qtype]
z = logits[0:k] / scale
p = softmax(z)
```

Buckets are `qtype:{2,3-5,6-10,11+}`. Temperatures are clamped to `[0.5, 5.0]`.

Answers (`04_MODEL_MAPPING.md`, upstream schema):

- **choice**: `choice = argmax(p)`, `probabilities`, and normalized Shannon
  confidence `1 - H(p)/log(k)` (Float32 evaluation, `k < 2` implies 1.0)
- **score**: zero-based expected value `sum(level * p)`, `legend`, probabilities,
  and the same entropy confidence
- **noul**: `noul = p[true]`, `confidence = max(p1, 1 - p1)`

The action head gives `action.act_probability = softmax(action)[0]`. The result
object carries `model`, per-question `answers`, and `usage`
(`input_tokens`, `output_tokens`). All reported values are rounded to 4 digits as
upstream does.

---

## 9. tenferro-infer requirements and mapping

Laya maps to the shared primitives (`11_TENFERRO_API_SURVEY.md` §5):

| Laya step | primitive |
|---|---|
| embedding gather | gather / `index_select` |
| LayerNorm | LayerNorm primitive (composed; fused later) |
| fused residual+norm | residual+norm primitive |
| QKV / MLP linear | prepared GEMM / `dot_general` |
| GeGLU | fused gate primitive (gelu-first-half × second-half) |
| exact GELU | GELU primitive (erf form) |
| RoPE | ModernBERT RoPE variant (pair-half, per-kind base) |
| full / sliding attention | reference attention + mask metadata |
| marker gather | gather |
| scorer / action / calibration | linear + elementwise; calibration on the host |

`tenferro-infer` must expose the ModernBERT RoPE variant and exact GELU; Laya
must not reimplement them.

---

## 10. CPU optimization plan

Following `03_CPU_GPU_INFERENCE.md`:

- weight packing and one canonical runtime layout
- preallocated workspace per class: embed/hidden A and B, QKV, attention
  output, MLP intermediate, head scratch, scorer/action scratch
- fused residual + normalization
- SIMD LayerNorm, GELU/GeGLU gate, masked softmax
- optimized attention with compact masks; sliding attention skips tiles outside
  the window
- thread policy that avoids nested outer-Rayon × BLAS oversubscription
- shape-bucketed prepared plans for common lengths (64/128/256/512)

Exit criteria for the milestone (`06_ROADMAP.md` Phase 3): reproducible report,
no avoidable large forward allocation, faster than the unoptimized reference,
correctness unchanged.

---

## 11. CUDA plan

- device-resident weights, RoPE tables and workspaces
- compact input upload and result download only
- fused/optimized attention with online softmax; sliding-window tile skipping
- device-resident decision head, scorer and action head
- explicit synchronization at boundaries; no silent CPU fallback

CUDA is followed immediately after the CPU milestone (`README.md`,
`06_ROADMAP.md` Phase 4).

---

## 12. Workspace and prepared plans

- Model owns immutable weights and plans; context owns mutable workspace.
- Plans are shape-bucketed; a generic fallback guarantees correctness.
- CUDA workspaces are context/stream-owned with asynchronous lifetime rules.

---

## 13. Testing and parity

Reference: Laya.jl CPU (and optional Metal/MLX), per
`05_TESTING_BENCHMARKS.md` §2.

- tokenizer goldens, prompt/marker goldens, serialization goldens
- intermediate activations: embeddings, embed norm, QKV, attention, residual,
  MLP, encoder final, decision head, marker values, logits, action
- shape regression: `L = 1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 512`,
  batch 1 and > 1, interior mask holes, one option and many options, local
  window boundaries
- semantic suite across all decision types; numerical tolerance documented per
  backend/dtype
- benchmark matrix `short:{1,10,50}`, `long:{1,10}` with p50/p95/min/max,
  allocations, peak memory, agreement

---

## 14. decision-core integration

`Agent` implements the shared `DecisionEngine` seam proposed in
`10_DECISION_ABSTRACTION.md`. Because upstream names the entry point
`system_one`, that name is the natural trait method (`04_MODEL_MAPPING.md` §4).
Answers are returned as `decision-core` `Answer` values; JSON/schema rendering
belongs to the caller or a thin adapter.

---

## 15. Open questions

- [ ] Exact JSON float formatting parity strategy (port `py_float` or use a
      vetted formatter with matching edge cases).
- [ ] Tokenizer asset formats to support (byte-level and Metaspace) and how much
      of the `tokenizers` behavior must be reproduced.
- [ ] Whether prompt building lives in `laya-infer` or `decision-core` given
      `DecisionEngine`'s `State` decision (`10_DECISION_ABSTRACTION.md` §4).
- [ ] Which fusions are worth dedicated extension ops (attention, GeGLU,
      residual+norm) versus composed tenferro ops.
- [ ] Whether to match MLX GELU/erf evaluation exactly on CPU or only within
      tolerance.
- [ ] Apple GPU sequencing given WebGPU's narrow coverage
      (`11_TENFERRO_API_SURVEY.md` §10).
