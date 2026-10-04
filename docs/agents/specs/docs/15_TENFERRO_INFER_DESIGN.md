# tenferro-infer Design

**Status:** design proposal (implements the package's shared primitive layer)  
**Date:** 2026-10-04  
**Related:** `01_DESIGN.md` §4, `09_JEFFCLIENT_ANALYSIS.md`, `11_TENFERRO_API_SURVEY.md`, `13_LAYA_INFER_DESIGN.md`, `14_JEFF_INFER_DESIGN.md`

`tenferro-infer` is the reusable inference-primitive crate shared by
`laya-infer` and `jeff-infer`. It is **not** a generic neural-network framework:
it exposes exactly the operations the two engines need, with reference and
optimized forms, on top of tenferro-rs.

---

## 1. Scope and responsibilities

Owns:

- linear / prepared GEMM helpers
- embedding gather
- LayerNorm and RMSNorm (centered and non-centered)
- GELU (erf-based) and GeGLU
- SiLU and gated SiLU
- stable softmax and masked softmax
- ModernBERT RoPE (non-traditional, pair-half) and Qwen partial RoPE
- residual + normalization composition and fusion
- reference multi-head attention
- inference layout helpers (head split/merge, packing)
- optional fused extension ops for the above

Does not own:

- model structure, checkpoint loading, tokenization (engine crates)
- Gated DeltaNet (in `tenferro-gated-delta`)
- backend/runtime construction (the caller owns the backend)
- autodiff or training rules

---

## 2. Crate layout and public API

```text
crates/tenferro-infer/
├── src/
│   ├── lib.rs
│   ├── linear.rs        # Linear / prepared GEMM
│   ├── embedding.rs     # gather
│   ├── norm.rs          # LayerNorm, RMSNorm (centered / non-centered)
│   ├── activation.rs    # gelu (erf), geglu, silu, gated silu
│   ├── softmax.rs       # stable / masked softmax
│   ├── rope.rs          # ModernBERT and Qwen variants
│   ├── attention.rs     # reference multi-head attention
│   ├── layout.rs        # head split/merge, packing helpers
│   ├── fusion.rs        # residual+norm, gate fusions
│   └── extension.rs     # optional extension-op wrappers
└── tests/
```

The API is ordinary Rust functions and small types that take a
`&mut dyn BackendSession` (plus the linalg trait for any triangular work) and
`Tensor`/views. No hidden global state; prepared plans and workspaces are
caller-owned.

```rust
pub struct RmsNorm { pub weight: Tensor, pub eps: f32, pub centered: bool }

pub fn rms_norm(session: &mut dyn BackendSession, x: &Tensor, p: &RmsNorm) -> Result<Tensor>;
pub fn layer_norm(session: &mut dyn BackendSession, x: &Tensor, p: &LayerNorm) -> Result<Tensor>;
pub fn gelu(session: &mut dyn BackendSession, x: &Tensor) -> Result<Tensor>;
pub fn silu(session: &mut dyn BackendSession, x: &Tensor) -> Result<Tensor>;
pub fn geglu(session: &mut dyn BackendSession, x: &Tensor, axis: usize) -> Result<Tensor>;
pub fn softmax(session: &mut dyn BackendSession, x: &Tensor, axis: usize) -> Result<Tensor>;
pub fn masked_softmax(session: &mut dyn BackendSession, x: &Tensor, mask: &Tensor, axis: usize) -> Result<Tensor>;
pub fn modernbert_rope(session: &mut dyn BackendSession, x: &Tensor, base: f32, positions: &Tensor) -> Result<Tensor>;
pub fn qwen_partial_rope(session: &mut dyn BackendSession, x: &Tensor, base: f32, rotary_dim: usize, positions: &Tensor) -> Result<Tensor>;
```

Prepared variants (`RmsNormPlan`, `AttentionPlan`) are added where profiling
justifies them.

---

## 3. Primitive specs

### 3.1 Linear / prepared GEMM

`y = x @ Wᵀ` (or `x @ W` depending on the chosen canonical layout). Prepared at
load time so the checkpoint's `(out, in)` layout is transposed/packed once. The
engine crates pass prepared weights.

### 3.2 Embedding gather

`embedding[:, ids]` with validation of index bounds. Implemented with `gather` /
`index_select`. Must not copy through the host on GPU.

### 3.3 LayerNorm and RMSNorm

- LayerNorm: mean/variance over the feature axis, `(x - μ) * rsqrt(σ² + eps) * w
  + b` (`b` optional). ModernBERT uses `eps = 1e-5`, no bias by default.
- RMSNorm: `x * rsqrt(mean(x²) + eps) * w`.
- `centered = true` uses `(1 + w)`; `centered = false` uses `w` directly. This
  distinction is required by Jeff (`09_JEFFCLIENT_ANALYSIS.md` §5).

Both are composed from `reduce_sum` / `reduce_sum_squares` / `rsqrt` / elementwise
ops; SIMD/fused forms are optimizations behind the same signature.

### 3.4 GELU and GeGLU

- GELU is the exact erf-based form `x * (1 + erf(x/√2)) / 2`, matching the MLX
  evaluation used by Laya (`13_LAYA_INFER_DESIGN.md` §7.1).
- GeGLU: for `y = [value; gate]` along an axis, `gelu(value) * gate` (Laya's
  order). The axis and half-split are parameters.

### 3.5 SiLU and gated SiLU

- `silu(x) = x * sigmoid(x)`.
- Gated SiLU: `silu(gate) * up` (Jeff MLP).
- `sigmoid` is provided as a small helper (`1/(1+exp(-x))` or the tanh form) so
  both engines use one definition.

### 3.6 Softmax and masked softmax

Numerically stable over the given axis: subtract the max, exponentiate, sum,
divide. Masked softmax sets masked scores to `-inf` (or `-floatmax`) before the
max. Used by Laya (column softmax) and Jeff (causal key softmax).

### 3.7 RoPE variants

- **ModernBERT** (non-traditional): within each head, rotate the pair
  `(i, i + head_dim/2)` by `position * base^(-2i/head_dim)`; base is per layer
  kind (`13_LAYA_INFER_DESIGN.md` §7.1).
- **Qwen partial**: rotate only the first `rotary_dim` channels; positions are
  absolute over the trimmed row (`09_JEFFCLIENT_ANALYSIS.md` §6).

Cos/sin tables are prepared at load time (per base/length, bounded cache owned by
the model). No per-request table generation.

### 3.8 Residual + normalization

`residual_norm(x, y, norm)` returns `(x + y, norm(x + y))`. Laya fuses this into
its layer boundary; the fused form must be reference-equivalent.

### 3.9 Attention

Reference multi-head attention: split Q/K/V, optional RoPE, scale, mask, stable
softmax, value product, merge heads. Optimized attention is added after
reference parity and profiling, and must support both full and sliding masks
using compact metadata rather than dense `L × L` matrices.

### 3.10 Layout helpers

Head split/merge, Q-feature/marker packing, and the layout conversions the model
crates share. These stay small and documented.

---

## 4. Reference-first policy

Every primitive follows `01_DESIGN.md` §3.3:

```text
correct tensor composition → reference parity → profiling → fusion / specialization
```

The reference forms are the normative semantics; fused forms must match them
within the documented tolerance.

---

## 5. CPU optimization

- SIMD for memory-bound/reduction kernels: LayerNorm, RMSNorm, GELU/SiLU,
  softmax, gated operations.
- GEMM delegated to the tenferro CPU provider (faer/BLAS), not hand-written.
- Residual+norm and gate fusions reduce allocations and passes.
- Thread policy coordinates with the provider to avoid nested parallelism.
- Workspaces are caller-owned and reused.

---

## 6. CUDA plan

- primitives map to tenferro CUDA ops (`dot_general`, reductions, elementwise,
  gather, slice, concatenate) where available
- fused kernels (attention, norm, RoPE, gates) are extension ops / `cuda::raw`
  kernels
- device-resident tables and workspaces; explicit transfer only at boundaries
- unsupported dtype/shape returns a typed error; no silent CPU fallback

---

## 7. Fused extension ops

Candidates (only when profiling proves the composed form is a bottleneck):

- residual + RMSNorm / LayerNorm
- masked softmax
- GeGLU / SiLU gate
- partial RoPE + head layout
- attention (sliding-window aware)

Each is an `ExtensionOp` with semantic parameters in its identity (axes,
normalization mode, etc.), bounded caches owned by the runtime, and no
process-global mutable state.

---

## 8. Numerical policy

- MVP `f32`; accumulate in `f32` unless a primitive explicitly requires F32
  accumulation for stability (norm statistics, softmax, Delta state).
- Exact GELU/erf evaluation matches the reference.
- `eps` placement is part of each primitive's contract (RMSNorm adds `eps`
  inside the square root).
- Documented `atol`/`rtol` per backend/dtype; no universal bit-identity.

---

## 9. Testing

- independent numerical tests per primitive against a CPU reference
- cross-backend parity (CPU vs CUDA) in the documented tolerance
- boundary shapes and dtypes
- fused vs reference equivalence
- CPU microbenchmarks for every major primitive

---

## 10. Open questions

- [ ] Canonical weight layout for `Linear` (`x @ W` vs `x @ Wᵀ`) and where the
      transpose happens.
- [ ] Which primitives should be prepared plans versus plain functions.
- [ ] Shared RoPE table cache ownership (model vs plan).
- [ ] Whether attention should expose a compact mask type or take validity +
      window separately.
- [ ] Fused-op identity details and cache keys.
- [ ] Whether `tenferro-infer` should expose a `sigmoid`/`silu` scalar path for
      host calibration reuse.
