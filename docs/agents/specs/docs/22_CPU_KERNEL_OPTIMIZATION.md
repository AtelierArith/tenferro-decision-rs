# CPU Kernel Optimization: BLAS / SIMD Opportunities

**Status:** analysis (priorities to be confirmed with the Jeff timings in `21_SPEED_COMPARISON.md`)  
**Date:** 2026-10-04  
**Context:** issue [#2](https://github.com/AtelierArith/tenferro-decision-rs/issues/2)

## Current state

- The host forwards (`forward_reference`) use hand-written triple loops
  (`linear_host` / `linear_into`) with no BLAS or explicit SIMD.
- The tenferro forwards delegate the heavy ops to `EagerSession`
  (`dot_general`, `rms_norm`, `attention`, …). `tenferro-cpu`'s default backend
  is **`faer`** (`cpu-faer`), so `dot_general`/`triangular_solve` already get an
  optimized, multi-threaded GEMM. This is why the Laya tenferro path (845 ms at
  L8B1) is ~3× faster than the host path (2692 ms) — but it still rebuilds every
  weight tensor per call.
- No workspace crate links BLAS/`matrixmultiply`/SIMD yet.

## Implemented (2026-10-04)

`cpu-kernels` now provides the host GEMM used by Laya, Jeff, and
`tenferro-gated-delta`:

- `matmul_row_major{,_into,_add_into}` — `y = weightᵀ x` for row-major
  `(in, out)` weights (Jeff, DeltaNet projections).
- `input_mul_weight_transpose{,_into,_add_into}` — `y = x weightᵀ` for the Laya
  `LinearWeights` layout.
- Backed by `matrixmultiply::sgemm`, parallelized across output rows with
  `rayon` above `PARALLEL_THRESHOLD` multiply-adds.

Measured effect (`21_SPEED_COMPARISON.md`): **12–30×** faster than the naive
host loops; a further 1.25–1.3× from `rayon` at larger shapes. All parity tests
still pass.

Still open below (attention `QKᵀ`/`PV`, SIMD elementwise, tenferro weight
caching, DeltaNet scan).

## A. GEMM (BLAS-class) candidates

| location | operation | notes |
|---|---|---|
| `laya-infer/src/model.rs::linear_host` | `(in,out)·(in,L)` projections: Wqkv, Wo, MLP Wi/Wo, head linear1/2, act1/2, scorer | the dominant matmul count (hidden 1024 × 28 layers) |
| `jeff-infer/src/model.rs::linear_host` | full-attention q/gate/k/v/o, MLP gate/up/down, readout | |
| `jeff-infer/src/model.rs::full_attention_host` | `QKᵀ` and `PV` | O(L²·hidden); matters for long context |
| `laya-infer/src/model.rs::attention_host` / `attention_block_host` | `QKᵀ`, `PV` | ModernBERT sliding/global windows |
| `tenferro-gated-delta/src/layer.rs::linear_into` | qkv, z, a, b, out_proj | 18 DeltaNet layers in Jeff |
| `tenferro-gated-delta/src/reference.rs::delta_scan_reference` (and `recurrent.rs`) | `state·k`, `S·q` (GEMV per token) | block the per-token loop |

Proposed: route these through one host GEMM helper. Best option is to reuse
**`faer`** (already in tenferro-cpu's tree, so no second BLAS and consistent
numerics), or add `matrixmultiply`. Alternatively enable tenferro-cpu's
`cpu-blas` + `blas-accelerate`/`blas-openblas` features.

## B. SIMD (elementwise / reduction) candidates

| location | operation |
|---|---|
| `laya-infer/src/model.rs::layer_norm_host`, `jeff-infer/.../rms_centered_rows`, `rms_heads`, `tenferro-gated-delta/ops.rs::rms_noncentered(_in_place)` | RMS/LayerNorm mean-square reductions |
| `laya-infer/.../attention_host`, `jeff-infer/.../full_attention_host` | softmax max/exp/sum |
| `jeff-infer/.../rope_partial_host`, `laya-infer/.../rope_host`, `tenferro-infer/rope.rs::rotate_pair_half` | RoPE rotation |
| `laya-infer/.../{exact_gelu,mlx_erf,relu,gelu_gate_host}`, `jeff-infer/.../sigmoid`, `gated-delta/ops.rs::{sigmoid,silu,softplus}` | activations |
| `tenferro-gated-delta/conv.rs::causal_depthwise_silu_into` | fused depthwise conv (vectorize over channels) |
| `tenferro-gated-delta/ops.rs::l2_normalize` | per-position Q/K normalization |
| `laya-infer/.../gather_markers_host`, `pool_host`, `tenferro-infer/embedding.rs` | gathers / pooling |

Proposed: use the `wide` crate (portable `f32x8` with CPU feature detection) or
`pulp`; parse-friendly strided loops can also rely on autovectorization. Avoid
`std::simd` (nightly-only at rustc 1.85). Thread across heads/batch with
`rayon` (already in tenferro-cpu).

## C. Tenferro-path overhead (not BLAS, but the largest eager win)

- `laya-infer/.../encoder_tensor`, `jeff-infer/.../full_attention_tenferro`,
  `tenferro-gated-delta/layer.rs::delta_layer_tenferro` rebuild weights with
  `host_to_tensor`/`constant` on **every call**, and bounce activations
  host↔tensor (`extract`/`duplicate_value`). Cache the weight tensors in a
  prepared plan/workspace (the DeltaNet plan/workspace added in `b3e739d` is the
  template) and keep activations tensor-resident.
- `tenferro-gated-delta/chunked.rs` extracts each chunk's output to host; keep
  it on the session and slice at the end. Host `cumsum`/`pair_decay` can be SIMD.

## D. Algorithmic

- Laya engine answers one question per row (B=1); add a batched forward
  (issue #3) to amortize weight reads.
- DeltaNet: choose recurrent vs chunked by shape (the plan already resolves
  this); benchmark both on Jeff and pick per shape bucket.

## Prioritization

1. GEMM for the host projections (A) — largest single win and simplest.
2. Hoist weight tensors / avoid host round-trips in the tenferro path (C).
3. Attention `QKᵀ`/`PV` GEMM (A) — matters as sequence length grows.
4. SIMD norms/activations/softmax/conv/RoPE (B).
5. DeltaNet recurrent scan blocking (A/B).

## Measured priorities (2026-10-04)

After the GEMM + rayon work (`21_SPEED_COMPARISON.md`):

- **Laya** is ~1.7–2.3× behind Julia, whose speed is almost entirely its
  multithreaded OpenBLAS. The remaining gap is BLAS-kernel quality
  (`matrixmultiply` vs OpenBLAS/Accelerate) rather than our elementwise code.
  Next lever: link a real BLAS (tenferro-cpu `cpu-blas` + `blas-accelerate`
  already in the tree) or `faer`.
- **Jeff** is 2.0× (L8) to 7.2× (L64) behind Julia's best (`-t 8`), and the gap
  grows with length. Julia wins here because of **fused task-parallel kernels
  with per-head workspaces** (`parallel_projections`, `parallel_full_heads`,
  `recurrent_delta`, `@turbo`), not BLAS. Next levers, in order:
  1. Fuse and parallelize the DeltaNet recurrent scan (`recurrent.rs`) and the
     elementwise/normalization passes with SIMD (`wide`) — the dominant L64 cost.
  2. Batch the attention `QKᵀ`/`PV` across heads and use GEMM.
  3. Reuse buffers (workspace) to cut allocations in the fused kernels.
