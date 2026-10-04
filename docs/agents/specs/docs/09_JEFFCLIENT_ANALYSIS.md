# JeffClient.jl Source Analysis

**Status:** analysis / implementation reference  
**Date:** 2026-10-04  
**Analyzed source:** `extern/JeffClient.jl` at commit `1338f4e7fc34a281e55d4ca5cf50565b9b91178b`

This document records the observed implementation facts of `JeffClient.jl`'s
native inference path, as the concrete reference for the recommended
`jeff-infer` / `tenferro-gated-delta` Rust crates. It complements
`04_MODEL_MAPPING.md` (high-level mapping) and `08_SOURCE_SNAPSHOT.md`
(repository inventory) by describing *how* the Julia engine actually computes.

`onnx.jl` and the ONNX export tooling are out of scope for the Rust port; only
the native safetensors path is described.

---

## 1. Backend architecture

`NativeBackend` is the model container. It is generic over the layer vector
type so that the forward loop stays concretely typed.

```julia
struct NativeLayer{A,M,V}
    attention::A      # :full or :delta
    mlp::M
    input_norm::V
    post_norm::V
end

struct NativeBackend{E,R,L,N,C}
    embedding::E
    readout::R
    layers::L
    final_norm::N
    config::C
    temperature::Float64
    max_options::Int
end
```

- Device is selected with `device = :cpu | :metal | :cuda`.
- GPU support is an optional package extension (`ext/JeffClientCUDAExt.jl`,
  `ext/JeffClientMetalExt.jl`); the core crate has no GPU dependency.
- Per-layer attention is a tagged NamedTuple with `kind = :full` or
  `kind = :delta`, using the checkpoint's `layer_types` array.
- CPU performance extensions (`LoopVectorization`, `AppleAccelerate`,
  `Octavian`, SIMD, …) hook into `cpu_portable_*` functions in
  `native_cpu.jl`; they do not change the algorithm.

This maps to the design package's split:

| Julia | Rust plan |
|---|---|
| `NativeBackend` / `NativeLayer` | `jeff-infer` model + layer types |
| `native.jl` reference kernels | `jeff-infer` reference path |
| `native_cpu.jl` + CPU extensions | CPU optimized path |
| `ext/cuda_*.jl` | CUDA path |
| `ext/metal_*.jl` | Apple GPU (later) |
| Delta algorithms in `native*.jl` / `ext/*_delta.jl` | `tenferro-gated-delta` |

---

## 2. Checkpoint and configuration

A checkpoint directory contains:

```text
config.json               # model_type == "qwen3_5", text_config, layer_types
decision_config.json      # format_version == 1, temperature, max_options
model.safetensors         # or model.safetensors.index.json
readout.safetensors       # "weight"
tokenizer assets (not used by the native prepared-token path)
```

Load-time validation (`NativeBackend(checkpoint)`):

- `source["model_type"] == "qwen3_5"`
- `decision["format_version"] == 1`
- `text_config["attention_bias"] == false`
- `text_config["hidden_act"] == "silu"`
- `rope_parameters["rope_type"] == "default"`
- `linear_num_value_heads % linear_num_key_heads == 0`
- `num_attention_heads % num_key_value_heads == 0`
- `1 <= max_options <= size(readout, 2) <= 255`
- `isfinite(temperature) && temperature > 0`

Runtime config derived from `text_config`:

```text
hidden       = hidden_size
head_dim     = head_dim
heads        = num_attention_heads
kv_heads     = num_key_value_heads
key_heads    = linear_num_key_heads
value_heads  = linear_num_value_heads
key_dim      = linear_key_head_dim
value_dim    = linear_value_head_dim
eps          = rms_norm_eps
rope_theta   = rope_parameters.rope_theta
rotary_dim   = head_dim * rope_parameters.partial_rotary_factor
```

Only `language_model.*` tensors are loaded. Layer order is given by
`text_config.layer_types` (`"full_attention"` / `"linear_attention"`), and
names are resolved by index, e.g. `language_model.layers.3.self_attn.q_proj.weight`.

---

## 3. Weight layout

`safetensors.jl` reverses every tensor's shape when loading, because safetensors
is row-major while Julia arrays are column-major. As a result native weights are
held as **(in_features, out_features)** and every linear is computed as

```julia
native_linear(weight, x) = transpose(weight) * x
```

Consequences for the Rust port:

- The safetensors file stores nn.Linear-style `(out_features, in_features)`.
- A naive port keeps `y = W @ x`; a load-time transpose/pack to `(in, out)`
  changes nothing semantically but changes the GEMM operand order.
- The design package's "load-time weight preparation" should record which
  layout is canonical at runtime.

### Tensor shapes

| Tensor | Julia shape (in, out) |
|---|---|
| `embed_tokens.weight` | `(hidden, vocab)` |
| `self_attn.q_proj.weight` | `(hidden, 2 * head_dim * heads)` |
| `self_attn.k_proj.weight` | `(hidden, head_dim * kv_heads)` |
| `self_attn.v_proj.weight` | `(hidden, head_dim * kv_heads)` |
| `self_attn.o_proj.weight` | `(head_dim * heads, hidden)` |
| `self_attn.q_norm.weight` | `(head_dim,)` |
| `self_attn.k_norm.weight` | `(head_dim,)` |
| `linear_attn.in_proj_qkv.weight` | `(hidden, 2*key_dim*key_heads + value_dim*value_heads)` |
| `linear_attn.in_proj_z.weight` | `(hidden, value_dim*value_heads)` |
| `linear_attn.in_proj_a.weight` | `(hidden, value_heads)` |
| `linear_attn.in_proj_b.weight` | `(hidden, value_heads)` |
| `linear_attn.conv1d.weight` | depthwise taps `(taps, channels)` |
| `linear_attn.A_log` | `(value_heads,)` |
| `linear_attn.dt_bias` | `(value_heads,)` |
| `linear_attn.norm.weight` | `(value_dim,)` |
| `linear_attn.out_proj.weight` | `(value_dim*value_heads, hidden)` |
| `mlp.gate_proj.weight` | `(hidden, intermediate)` |
| `mlp.up_proj.weight` | `(hidden, intermediate)` |
| `mlp.down_proj.weight` | `(intermediate, hidden)` |
| `input_layernorm.weight` | `(hidden,)` |
| `post_attention_layernorm.weight` | `(hidden,)` |
| `norm.weight` | `(hidden,)` |
| readout `weight` | `(hidden, options)` |

---

## 4. Forward pass overview

`logits(backend, inputs)` requires `input_ids` and `attention_mask`, both
logical `(batch, sequence)`. Constraints:

- token IDs in `[0, vocab)`
- mask values in `{0, 1}`
- the final position of every row must be active (left padding)
- batches are processed row-by-row (a shared/batched path is experimental)

Per row:

```text
first  = first active position (leading zeros skipped; CPU default, CUDA opt-in)
hidden = embedding[:, ids[first:end] + 1]        # (hidden, L)
hidden = hidden_forward(layers, mask[first:end], final_norm)
scores = readout' * hidden[:, end]               # (options, 1)
```

`hidden_forward` is a normal pre-norm stack:

```text
for each layer:
    normalized = RMSNorm(x, input_norm)
    mixed      = attention(normalized)
    residual   = x + mixed
    normalized = RMSNorm(residual, post_norm)
    residual  += MLP(normalized)
    x          = residual
x = RMSNorm(x[:, end], final_norm)
```

Note: because leading masked positions are trimmed before the forward pass,
RoPE positions restart at zero at `first`. Interior mask holes are preserved.

---

## 5. RMSNorm semantics

The single norm helper has a `centered` flag that changes the weight term:

```julia
native_rms(x, weight, eps; centered = true)
    scale = 1 / sqrt(mean(x^2) + eps)          # mean over feature dim
    out   = x * scale * (centered ? (1 + weight) : weight)
```

- **centered = true** (`1 + weight`): input/post/final norms, and Q/K norms.
- **centered = false** (`weight`): the DeltaNet output norm.

This distinction is load-bearing and must not be collapsed in the Rust
implementation. `eps` is added inside the square root, before inversion.

---

## 6. Full attention

```text
qgate = reshape(q_proj' x, 2*head_dim, heads, L)   # first half query, second half gate
q = RoPE(RMSNorm(qgate[1:head_dim], q_norm))
k = RoPE(RMSNorm(k_proj' x reshaped, k_norm))
v = v_proj' x reshaped
mask[i, j] = (i <= j && mask[i] == 1) ? 0 : -floatmax(Float32)
for head:
    scores = (k_kv' q_head) / sqrt(head_dim) + mask
    probs  = softmax(scores, dim=1)                 # keys axis
    out_head = v_kv * probs * sigmoid(qgate_gate_head)
out = o_proj' out
```

Key details:

- **Q carries a gate**: `q_proj` emits `2 * head_dim` per head; the second half
  is passed through `sigmoid` and multiplies the attention output.
- **GQA**: query head `h` maps to KV head `ceil(h / (heads / kv_heads))`
  (consecutive grouping).
- **Partial RoPE**: applies to the first `rotary_dim` channels of `head_dim`.
  For `half = rotary_dim / 2`, channels `(i, i+half)` are rotated by
  `angle = position * rope_theta^(-2i/rotary_dim)`. Remaining channels pass
  through unchanged.
- Positions are absolute indices `0 .. L-1` over the (possibly trimmed) row.
- Causal mask combines key-validity (`mask[key] == 1`) with `key <= query`.

---

## 7. Gated DeltaNet (`linear_attention`)

### 7.1 Preparation

```text
masked = x * mask                                  # zero invalid positions
mixed  = SiLU(causal_depthwise(qkv' masked))       # depthwise causal conv1d
q = L2-normalize(mixed[1:key_width]) * sqrt(key_dim)
k = L2-normalize(mixed[next key_width])
v = mixed[2*key_width + 1 : end]
z = SiLU(z_proj' masked)                            # gate (pre-activated)
beta  = sigmoid(b_proj' masked)
decay = (-exp(A_log)) * softplus(a_proj' masked + dt_bias)
```

where `key_width = key_dim * key_heads`, and L2 normalization is
`v / sqrt(sum(v^2) + 1e-6)`. `q` additionally scales by `sqrt(key_dim)`.

`causal_depthwise` is a depthwise causal convolution of width `taps`
(`kernel = size(weight, 1)`): output channel `c`, position `t` is
`sum_tap input[c, t - (taps - tap)] * weight[tap, c]`, followed by SiLU. It is
equivalent to a causal conv1d with no bias.

### 7.2 Recurrence formulation (reference, and optimized on CUDA/Metal)

Per value head, with state `S` of shape `(value_dim, key_dim)`:

```text
S = 0
for t in 1..L:
    factor = exp(decay[t])
    pred   = S * k[:, t]
    corr   = beta[t] * (v[:, t] - factor * pred)
    S      = factor * S + outer(corr, k[:, t])
    result = S * q[:, t]
    out[:, t] = RMSNorm(result, norm; centered=false) * z[:, t]
```

Here `q`, `k` are the key head mapped from the value head via
`ceil(head / (value_heads / key_heads))`. The RMS norm is non-centered.
This is the natural reference implementation and the CUDA/Metal hot path
(register/shared-memory state, one warp per value row).

### 7.3 Chunked formulation (CPU default and fallback)

Chunk size defaults to 64 (`delta_chunk_size`). For a chunk `[start, span]`
of length `n`:

```text
cumulative = cumsum(decay[span])                    # (n,)
pair_decay[i, j] = i >= j ? exp(cumulative[i] - cumulative[j]) : 0

weighted = k * beta
system   = (weighted' * k) .* pair_decay            # unit lower triangular

values_rhs = (v' .* beta)                           # (n, value_dim)
keys_rhs   = (weighted' .* exp(cumulative))         # (n, key_dim)
solve unit-lower-triangular system for both RHS     # trsm / UnitLowerTriangular

corrections = values_rhs' - S * keys_rhs'
intra       = (k' * q) .* pair_decay'
result      = S * (q .* exp(cumulative)) + corrections * intra
ending_keys = k .* exp(cumulative[n] - cumulative)
S           = corrections * ending_keys' + exp(cumulative[n]) * S

out = RMSNorm(result, norm; centered=false) * z
```

- The triangular solve is a **unit lower-triangular** solve (`trsm` on CPU,
  `UnitLowerTriangular \` in Julia). Metal falls back to the chunked path for
  `key_dim > 256`; a finite-series inverse was numerically unstable.
- Algorithm selection (`recurrent` vs `chunked`) must be deterministic per
  model/config/runtime, matching the design specification.

### 7.4 Output projection

Both formulations assemble `(value_dim * value_heads, L)` and apply
`out_proj'`.

---

## 8. MLP

```text
down_proj' ( SiLU(gate_proj' x) .* (up_proj' x) )
```

CUDA packs `gate`/`up` into a single concatenated matrix (`PackedMLP`) so one
GEMM produces both, then a fused gate kernel. SiLU is `x * sigmoid(x)`.

---

## 9. Readout and decision semantics

`logits` returns raw `(batch, max_options)` scores. `decide` then applies,
per question (from `questions.jl`):

```text
p = softmax((logits - max(logits)) / temperature)    # Float64, active options only
```

- **ChoiceQuestion**: `choice = argmax`, and
  `confidence = clamp((p_best - 1/n) / (1 - 1/n), 0, 1)`.
- **NoulQuestion** (yes/no, columns ordered false then true): returns
  `noul = p[true]`.
- **ScoreQuestion** (2..10 ordered levels): returns
  `score = sum(level * p)`, zero-based expected value, and a distance-based
  confidence.
- Option order is fixed by the caller and must match checkpoint column order.

---

## 10. CPU/GPU execution and workspaces

### CPU (`native_cpu.jl`, `cpu_settings.jl`)

- An automatic, immutable policy chooses between portable and Accelerate paths
  based on Apple Accelerate availability. It is not environment-driven.
- `parallel_projections` / `projection_thread_scope` manage nested BLAS/thread
  parallelism.
- Reusable workspaces: full-attention buffers (`qgate`, `q`, `k`, `v`, `out`,
  mask, cosine/sine tables, per-worker head scratch), Delta buffers (chunk
  matrices, RHS, corrections), MLP buffers, and normalization buffers.
- Fusions: MLP residual accumulation into the down projection (`beta = 1`),
  in-place Delta RMS, final-token-only MLP, and final-query-only last-layer
  attention.
- Depthwise conv, RMS, softmax, and Delta loops use column-major `@simd` loops,
  with optional extension hooks for vector math.

### CUDA (`ext/cuda_*.jl`)

- `ForwardWorkspace` is keyed by a weak reference to the model, owns a stream,
  and recycles slot-indexed scratch buffers; forwards sharing a model are
  serialized and synchronized at scope boundaries.
- `cublasGemmEx` is called directly with model-owned device scalars.
- Kernels: warp-reduction Q/K normalize, register-resident Delta recurrence
  (one warp per value row), causal depthwise conv + SiLU, warp softmax,
  prepare-head (RMS + partial RoPE + gate layout), value layout, merge gate,
  RMS, gather, packed MLP gate.
- Strided-batched GEMM handles per-head attention products.
- `JEFF_CUDA_TRIM_PADDING=1` opts into leading-padding trim; default keeps the
  full sequence for comparable benchmarks.

### Metal (`ext/metal_*.jl`)

- Fused SIMD recurrent Delta kernel for `key_dim <= 256`; chunked triangular
  solver otherwise.
- Fused RMS/L2 normalize, causal masked softmax, partial RoPE + head layout,
  output layout + sigmoid gate; queue-local RoPE tables; product-only MPSGraph.

---

## 11. Supported scope and non-goals

Supported by the native path:

- text-only Qwen3.5 hybrid (full attention + Gated DeltaNet)
- float32, bias-free projections, default (partial) RoPE
- prepared token IDs and attention mask; left padding
- batch processing (row-wise by default)

Not supported / out of scope:

- tokenizer inside the native API (tokens are prepared externally)
- generation, KV cache, training, image inputs
- quantized or reduced-precision weights
- non-default RoPE variants, attention bias

---

## 12. Implications for the Rust / tenferro port

1. **Two RMSNorm modes.** Keep centered and non-centered variants explicit;
   the checkpoint's `1 + weight` convention is easy to get wrong.
2. **Weight layout is a load-time decision.** Choose and document a canonical
   runtime layout, and treat the safetensors `(out, in)` layout as input only.
3. **Q gate.** `q_proj` doubles the per-head width; the gate half must be split
   before attention and multiplied after the value product.
4. **Partial RoPE.** Only `rotary_dim` channels rotate; positions are absolute
   over the trimmed row. Precompute frequency/cosine/sine tables at load time.
5. **GQA grouping** is consecutive (`ceil`), not strided; preserve exactly.
6. **DeltaNet needs a triangular solve** for the chunked path. If tenferro has
   no unit-lower-triangular solve primitive, provide it as a reference loop or
   an extension op, and keep the recurrent formulation as the correctness
   reference.
7. **Decay/beta/gate are elementwise precomputations** over projections and are
   cheap candidates for fusion; `exp(A_log)` is computed once at load time.
8. **Masking semantics.** Delta pre-multiplies by mask; full attention uses a
   causal + key-validity mask. Trimming resets positions; interior holes are
   kept. Encode this as compact metadata, not dense matrices.
9. **Final-token optimization** only computes the last position's MLP (and
   optionally last-layer query). It must be validated separately, since it is a
   semantic shortcut relying on the readout reading only the last token.
10. **Workspace ownership.** Model holds immutable weights; context/workspace
    holds scratch; CUDA context owns a stream. No process-global mutable cache.

---

## 13. Open items to verify before implementation

- [ ] Exact `layer_types` pattern for the 0.8B checkpoint (full/linear interleave).
- [ ] tenferro primitive coverage for gather, strided/batched GEMM, reductions,
      cumsum, softmax, and triangular solve.
- [ ] Whether tenferro's CUDA backend exposes enough control to keep per-head
      Delta state on-chip.
- [ ] Whether `rotary_dim == head_dim` full-RoPE variants must also be handled.
- [ ] Numerical tolerance policy against the PyTorch reference, and against the
      Julia native CPU path.
- [ ] Whether the Rust engine targets only prepared tokens (as Julia does
      natively) or also embeds a tokenizer (not required by the current spec).
