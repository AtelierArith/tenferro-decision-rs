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

## tenferro eager `dot_general` CPU path (2026-10-04)

Call path (pinned rev `471c4278`):

- `EagerRuntime::with_cpu_backend(CpuBackend::new())` — cpu threads follow
  `RAYON_NUM_THREADS` (`tenferro-cpu/src/context.rs`).
- `EagerSession::dot_general` → `EagerTensor::nary_op_in_session`
  (`tenferro-ad/src/eager_ops.rs:214`); inference takes the `!any_requires_grad`
  branch, so no AD is recorded.
- `exec_standard_op_on_tensor_reads_with_session` (`tenferro-ad/src/eager_exec.rs:358`)
  → `exec.dot_general_read` (`:451`).
- `CpuExecSession::dot_general_read` (`tenferro-cpu/src/exec_session.rs:650`)
  → `execute_dot_allocated` (`:557`): `preflight_dot_general`, pool
  `UninitTensor::acquire`, `DotGeneralRuntime::execute_dot_into_uninit`
  (`dot_runtime.rs:1044`); on decline, a zeroed pool alloc + scoped fallback.
- Provider: `builtin_gemm_provider(CpuBackendKind::default_compiled())` —
  **`Blas` when the `cpu-blas` feature is on, else `Faer`**
  (`tenferro-cpu/src/backend.rs:286`). We build with default features → **faer**.
- faer path: `FaerGemmProvider` (`provider.rs:2224`) →
  `faer::MatRef::from_raw_parts` + `faer::linalg::matmul::matmul_with_conj(...,
  Par::rayon(n))`, `n` = the cpu thread budget (`provider.rs:561`); below
  `FAER_PARALLEL_MIN_MULADDS = 1<<20` it forces `Par::Seq`.

Per-call overheads vs our `cpu-kernels`: output pool allocation (possible zero
fill), two `Box` allocations per operand (`promote_read_to_dtype`), a
`Vec<TensorRead>` per op, the GEMM analysis recomputed every call
(`cache_slot = None`, `exec_session.rs:656`), and a faer `spindle`
scope+barrier fan-out per GEMM (not the pool's steady state). For tall-skinny
shapes (`out_dim` small) the fan-out fraction is larger.

Levers that need **no edit to the pinned repo**:

- `tenferro-cpu` features: `cpu-faer` (default), `cpu-blas`, `provider-src`,
  `blas-accelerate`, `blas-openblas`, `blas-mkl`. Enabling `blas-accelerate`
  flips `default_compiled()` to Accelerate's `cblas_sgemm`.
- `cpu-kernels` (host path): (A1) Accelerate via `blas-src`+`cblas-sys`;
  (A2) faer with `Par::rayon(0)`; (A3) finer row/column blocking than the
  current `out_dim`-row split.

**Tried:** enabling `blas-accelerate` flips `default_compiled()` to `Blas`, but
`triangular_solve` then fails with `CPU linalg provider Blas is not compiled in`
(the BLAS provider has no CPU linalg kernels), so the DeltaNet cannot run. The
tenferro path must stay on faer until the BLAS provider covers linalg.

**Measured (MWE `bench-suite/examples/eager_dot_general_mwe.rs`):** the eager
`dot_general` wrapper adds only ~1.04–1.17× over the *same* faer GEMM run
directly (preallocated output, `Par::rayon(0)`), i.e. the wrapper cost is small.
For these projection shapes faer was even **faster** than our host
`matrixmultiply`/Accelerate kernel (~1.3–2.1×), so eager's GEMM provider is not
the whole-model bottleneck. The two fixable items are the per-call analysis
(cache slot unused) and the BLAS-without-linalg limitation (filed as #1992).

## Our usage: calling the fastest CPU kernel through the session

Since tenferro-rs itself is pinned, the wins that need no upstream change come
from how **we** use it. Two self-hosted extension ops put our host CPU kernels
behind the eager session, so the forward stays tenferro-first:

- `tenferro-gated-delta::extension::GatedDeltaOp` — the fused host recurrent
  Gated DeltaNet. **Now the Jeff tenferro default** (`DeltaKernel::HostRecurrent`;
  `TensorNative` remains for backend portability). Zero-copy: the op reads the
  host kernel's row-major operands from the column-major input buffers in place
  and borrows them (`GatedDeltaWeightSlices`). The old `(in, out)` column-major
  input layout transposed every weight per call (3–11× slower than the kernel);
  the row-major input layout removes it. Layer effect (`bench_delta_paths`,
  L8/16/64): extension op 1.0–1.13× the bare kernel and ~1.8× faster than the
  tensor-native chunked layer.
- `tenferro-ext::gemm::GemmOp` — a dense `y = weightᵀ x` projection backed by
  `cpu-kernels` (Accelerate). Correct and zero-copy, but **not** wired in: for
  our tall-skinny decode shapes faer (tenferro's default) is already faster
  (see the MWE above), so this op only proves the mechanism.

Whole-model effect (`bench_tenferro_kernels`, production Jeff checkpoint,
`RAYON_NUM_THREADS=8`, release, best-of-10):

| length | host | tenferro `HostRecurrent` | tenferro `TensorNative` |
|---:|---:|---:|---:|
| 8 | 169 ms | 292 ms (1.73×) | 344 ms (2.04×) |
| 16 | 224 ms | 360 ms (1.60×) | 459 ms (2.04×) |
| 64 | 739 ms | **629 ms (0.85×)** | 894 ms (1.21×) |

So `HostRecurrent` is consistently ~1.2–1.2× faster than `TensorNative` and
brings the tenferro path within ~1.7× of the host at L8 and ahead of it at L64,
all without touching tenferro-rs.

**Unified `(length, hidden)` orientation (our-usage lever).** The Jeff tenferro
forward was re-oriented to `(length, hidden)` end-to-end (embedding gather,
rms_norm over the last axis, `linear` contracting the last axis), removing the
per-layer `(hidden, length) ↔ (length, hidden)` transposes and making the
`GatedDelta` op's `(length, hidden)` output zero-copy. It is correct (all model,
fixture, and extension tests pass) and removes redundant ops, but the measured
wall-clock effect is **within noise** — a sampling profile shows
tenferro-cpu's `structural::typed_copy_into_uninit` barely moves (13898 → 13167
self samples at L8), so our forward transposes were not the bulk of the layout
copies; those come from elsewhere (norm broadcasts, RoPE, attention, the
`dot_general` path). The remaining L8 gap is faer's skinny-GEMM microkernels,
Accelerate (the host DeltaNet kernels), and thread-pool/barrier waits — small
-shape parallel inefficiency, not Rust bookkeeping. `TensorNative` gains two
transposes (it is written for `(hidden, length)`), so it is slightly slower than
before; it only matters off-CPU.
