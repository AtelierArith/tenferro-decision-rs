# CPU Kernel Optimization: BLAS / SIMD Opportunities

**Status:** CPU optimizations implemented; remaining performance differences measured in `21_SPEED_COMPARISON.md`
**Date:** 2026-10-04  
**Context:** issue [#2](https://github.com/AtelierArith/tenferro-decision-rs/issues/2)

## Initial baseline (before the optimizations below)

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

The later sections track attention `QKᵀ`/`PV`, SIMD elementwise, tenferro weight
caching, and the DeltaNet scan.

## Host-optimized forward (`host_opt`, Jeff) — 2026-10-04

`forward_reference` (the oracle) is unchanged; a separate optimized host path
was added in `jeff-infer/src/host_opt.rs` (`forward_host_opt{,_with}` +
`HostOptWorkspace`, selectable as `JeffBackend::HostOpt`). It ports the Julia
native-CPU techniques that the oracle lacks:

- **rayon at the right granularity**: the oracle parallelizes only inside GEMMs
  and the DeltaNet head loop. `host_opt` also parallelizes RMSNorm over tokens,
  full attention / head-norm / RoPE over heads, and the SiLU gate / residual adds
  over elements.
- **RoPE tables precomputed once per length** (the oracle recomputes
  `theta.powf(...)` + `cos`/`sin` per `(head, token, channel)`).
- **Reusable activation buffers** (`HostOptWorkspace`), so a warmed forward is
  allocation-free; the oracle allocates a `Vec` per intermediate and copies the
  DeltaNet output.

Parity: `host_opt` matches the oracle and the production reference to
`max diff ~1.2e-5` (scale 10.4) on the `mstrasser/Jeff-Qwen3.5-0.8B` checkpoint
(`real_checkpoint`), and the small-model tests (`tests/model.rs`,
`tests/engine.rs`) cover mixed/delta-only/full-only stacks.

Two shared mixed-precision-hotspots in `tenferro-gated-delta` were also fixed
(these help the oracle too; its source is untouched and its output is
**bit-identical**):

- `conv.rs::causal_depthwise_silu_into` was a serial triple loop over
  `channels=6144`; it is now parallel over channels. Delta-layer `conv+silu`
  stage: **4.0 → 0.8 ms** at L64.
- `recurrent.rs::scan_head` stored the state row-major `(value_dim, key_dim)` and
  did a scalar reduction with a stride-`length` gather of `k`/`q`. The state is
  now stored transposed `(key_dim, value_dim)` with the update and readout fused
  into one contiguous axpy over value dims (the same formulation as Julia's
  `cpu_delta_recurrent_heads!`). Delta layer: **14.8 → 11.6 ms** at L64.

Whole-model effect (production Jeff, `RAYON_NUM_THREADS=8`, release, best-of-15):

| length | oracle `forward_reference` | `host_opt` | host_opt/oracle | Julia best |
|---:|---:|---:|---:|---:|
| 8 | 161.7 ms | 171.2 ms | 1.06× | 137.5 ms |
| 16 | 184.4 ms | 178.9 ms | 0.97× | 143.3 ms |
| 64 | 466.3 ms | **347.4 ms** | **0.74×** | 238.3 ms |

So `host_opt` is ~1.34× faster than the oracle at L64 (within noise at L8, where
the rayon fan-out offsets the elementwise win) and ~1.24–1.46× behind Julia.

### The GEMM backend is *not* the remaining gap

Measured `cblas_sgemm` directly on the model shapes (best-of-30): Accelerate
(`cpu-kernels`) and Homebrew OpenBLAS 0.3.34 are within a few percent, e.g.
`(1024, 4096, len 64)`: Accelerate 1.42 ms vs OpenBLAS(8) 1.38 ms;
`(1024, 2048, len 8)`: 0.36 vs 0.22 ms. So swapping Accelerate for OpenBLAS
would not close the gap (unlike Laya, whose work is almost pure BLAS).

### The chunked DeltaNet is not the lever on x86

Julia's `cpu_defaults(apple)` sets `portable = !apple`, and `apple` requires
`Sys.ARCH === :aarch64`. On our **x86_64** machine `apple = false`, so Julia runs
`recurrent_delta = true` (chunk size 1) — the *same recurrent scan* we use.
Chunked (`TensorNative`) is Apple-Silicon-only and is ~1.8× *slower* here. So
porting a host chunked kernel would not explain or close the x86 gap; the
remaining ~1.4× is the distributed cost of the recurrent scan's per-token state
passes and the elementwise/transcendental work (SiLU `exp`, RMS `sqrt`), all of
which Julia pays too but hides better across its 8 task workers.

`JeffEngine` now defaults to `JeffBackend::Auto`, which uses `host_opt` (the
fastest at every measured length). `Host` (the oracle) and `Tenferro` remain
selectable.

### Explicit SIMD via `pulp`

Autovectorization did not fire for the DeltaNet scan's inner axpy loops (they
compiled to scalar `mulss`/`addss`), and `-C target-cpu=native` did not help
because the model build targets baseline x86-64. The scan token loop is now
dispatched through **`pulp`** (`recurrent.rs::ScanOp`), which emits
AVX2/FMA kernels and selects them at runtime via `Arch::new().dispatch(...)`, so
a baseline-compiled binary still gets SIMD. `pulp` was already in the tree (via
`faer`). Its isolated effect on the scan token loop is **1.83×**
(0.344 → 0.188 ms/head, best-of-200). The L64 delta layer dropped from 11.6 ms to
**8.5 ms**.

### Row-blocked `matrixmultiply` beats Accelerate for Jeff's layout

Counter to the earlier "Accelerate for large GEMMs" choice, `cpu-kernels`'s
rayon **row-blocked `matrixmultiply`** path is faster than Accelerate for the
Jeff/DeltaNet/MLP projections. Those use `matmul_row_major` (`y = Wᵀ x`), which
blocks over `out_dim` — large for every projection (1024–6144) and `length` is
small, so the row blocks fill the cores while Accelerate's per-call
`dispatch_apply` has high overhead. Back-to-back (`RAYON_NUM_THREADS=8`,
best-of-12): `host_opt` L8 **108** vs 153 ms, L16 **126** vs 160, L64 **250** vs
283. So `matmul_row_major` no longer calls BLAS.

Laya's kernel is `input_mul_weight_transpose` (`y = x·Wᵀ`). It originally
blocked over `rows` (= batch×sequence) — only 8–64 at decode — so the rayon path
could not parallelize and Accelerate won. It now **blocks over the large
`out_dim` axis instead** (each rayon task writes a disjoint column block of `y`
via strided `sgemm`), so it parallelizes at every shape. Laya L8B1: **106 → 93
ms** (1.19× → 1.05× of Julia); L64/L8B8 are within noise. This also makes the
`tenferro-ext::gemm::GemmOp` extension fast. With both kernels off BLAS,
`cpu-kernels` **drops the `cblas-sys`/Accelerate dependency entirely and is
portable**.

Whole-model effect (production Jeff, `RAYON_NUM_THREADS=8`, best-of-15):

| length | oracle | `host_opt` | host_opt/Julia | Julia best |
|---:|---:|---:|---:|---:|
| 8 | 126.5 ms | **106.8 ms** | **0.78×** | 137.5 ms |
| 16 | 137.1 ms | **124.3 ms** | **0.87×** | 143.3 ms |
| 64 | 379.6 ms | **248.7 ms** | **1.04×** | 238.3 ms |

`host_opt` is now **faster than Julia at L8/L16** and within ~4% at L64. The
remaining L64 gap is the last bit of scan/elementwise/MLP-silu overhead.

### Feeding the findings back to the tenferro-first path

The DeltaNet improvements live in shared crates, so they reach the tenferro
`GatedDelta` extension op (`DeltaKernel::HostRecurrent`) automatically: the
fused `tenferro-gated-delta::recurrent` kernel (`pulp` SIMD scan, transposed
state, parallel conv) and `cpu-kernels` projections are exactly what the
extension op executes. Measured `bench_tenferro_kernels`, `RAYON_NUM_THREADS=8`,
best-of-12: `HostRecurrent` **265 / 304 / 416 ms** at L8/L16/L64 (was ~293 / 351
/ 599 before the kernel work).

The **general** GEMMs (MLP, full attention) in the tenferro forward still go
through `linear` → `dot_general` → faer. Routing them through the self-hosted
`GemmOp` was prototyped (wrap `linear` in a transpose pair, since the op
contracts the first axis) and is a **net loss** here: the eager extension-op
dispatch plus the two transposes cost more than the kernel gain — measured
`HostRecurrent` L8 **345 vs 257 ms**, L64 **534 vs 409 ms**. So the tenferro
forward keeps faer for those; `GemmOp` stays available but unwired.

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
  tensor-native chunked layer. The op also reuses its `GatedDeltaWorkspace`
  across calls through the runtime's `ExtensionCacheStore`
  (`ExtensionCacheKey`, keyed by length), so the scratch is not reallocated per
  call.
- `tenferro-ext::gemm::GemmOp` — a dense `y = weightᵀ x` projection backed by
  `cpu-kernels` (Accelerate). Correct and zero-copy, but **not** wired in: for
  our tall-skinny decode shapes faer (tenferro's default) is already faster
  (see the MWE above), so this op only proves the mechanism.

### GEMM provider is shape-dependent, not the gap

> **Update (below):** this compared Accelerate against *faer*. It did not test
> the rayon row-blocked `matrixmultiply` path, which turns out to be faster than
> Accelerate for Jeff's `y = Wᵀ x` projections (see "Row-blocked
> `matrixmultiply` beats Accelerate"). The conclusion that the provider is *not*
> the tenferro-vs-host gap still holds for the tenferro path, which uses faer.

`bench-suite/examples/host_gemm_providers.rs` compares `cpu-kernels`
(Accelerate) against direct faer on the projection shapes (best-of-50, 8
threads):

| in | out | len | host (Accelerate) | faer | faer/host |
|---:|---:|---:|---:|---:|---:|
| 1024 | 4096 | 8 | 0.975 ms | 0.748 ms | 0.77× |
| 2048 | 1024 | 8 | 0.462 ms | 0.355 ms | 0.77× |
| 4096 | 1024 | 8 | 1.001 ms | 0.723 ms | 0.72× |
| 1024 | 4096 | 64 | 1.760 ms | 2.447 ms | **1.39×** |
| 4096 | 1024 | 512 | 8.139 ms | 16.806 ms | **2.06×** |

faer wins only for the very skinny decode shapes (`len = 8`) and **loses** at
`len = 64/512`, so switching `cpu-kernels` wholesale to faer would trade a ~3%
decode win for a large prefill regression. There is no single provider that
wins everywhere, and the difference is small at decode — so the provider is not
the tenferro-vs-host gap. Do not change it.

### What is *not* the L8 gap (summary)

The L8 gap is not the eager wrapper (~1.1×), not the GEMM provider
(shape-dependent, ~equal), not our forward transposes (neutral, above), and not
the DeltaNet kernel (fixed). It is the aggregate of the many small eager ops,
faer's per-call `spindle` scope/barrier fan-out for `m = length = 8` GEMMs, and
the memory traffic of the intermediates — all tenferro-internal. Capturing more
requires changes inside tenferro-rs (out of scope); the profile is filed
upstream as [#1995](https://github.com/tensor4all/tenferro-rs/issues/1995).

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

## Parallel Laya GeGLU (2026-10-08)

The fused `tenferro-ext::GegluOp` previously evaluated every token serially.
Its erf polynomial performs many multiply-adds per element, so the CPU kernel
now distributes independent token columns across Rayon once the output has
at least 16,384 elements. Small shapes retain the serial path. The formula and
per-element evaluation order are unchanged; a regression test checks exact
floating-point bits against the scalar formula with one and three workers and
uneven feature/column sizes.

On Ryzen 9 PRO 8945HS, eight Rayon threads, float32, warmup 2 and median of 5,
cached tenferro Laya L64 B1 improved from 297.3 to 220.2 ms and L8 B8 from
284.5 to 211.3 ms (about 26% shorter). An intermediate rerun measured 223.5
and 243.7 ms, respectively; the batching result varies more between runs.
The host oracle is untouched. This improves the production tenferro path but
does not eliminate its remaining Julia/Python gap.

Two GEMM alternatives were also measured and rejected: fewer output-column
tasks improved isolated contractions but had no consistent model-level gain;
faer was faster on some square projections and slower on the wide MLP
projection. Neither experimental change is included.

## Remaining Laya CPU cost (2026-10-08)

Temporary `Instant` timers on the cached tenferro encoder and its GEMM
extension measured the remaining cost after parallel GeGLU. Each encoder
stage below is the median of five per-forward sums across all 28 layers,
excluding two warmup forwards. Ryzen CPU, float32, eight Rayon threads.

| shape | attention + projections + residual | MLP norm | MLP up projection | GeGLU | MLP down + residual |
|---|---:|---:|---:|---:|---:|
| L8 B1 | 26.3 ms | 0.7 ms | 30.5 ms | 1.8 ms | 13.0 ms |
| L64 B1 | 67.0 ms | 4.1 ms | 67.0 ms | 9.7 ms | 29.6 ms |
| L8 B8 | 56.3 ms | 4.0 ms | 66.6 ms | 9.4 ms | 29.2 ms |

These are diagnostic timings, not a replacement for uninstrumented latency.
Embedding, pre-attention norm, mask construction, final norm, and decision/action
heads are outside these windows. Reports and timer definitions are recorded in
`fixtures/bench-geglu-cpu-2026-10-08/laya_stage_profile.json`; the instrumentation
was removed afterwards.

No GEMM call took the `to_contiguous_read` fallback. The large MLP projection's
kernel median was about 1.9 ms versus about 2.4 ms for the eager projection
call. Removing fallback materialization therefore cannot explain the remaining
gap. Both kernel execution and eager-call overhead matter.

An explicit runtime-selected x86 FMA version preserved scalar GeGLU bits but
changed full-model L64 B1 only from 220.2 to 215.9 ms, within the observed
variation. A full-model faer experiment at 32 or more rows measured L64 B1
231.9 ms and L8 B8 210.4 ms versus 220.2/211.3 ms with the existing kernel;
its improvements to the host oracle did not carry over to the production
tenferro path. Both changes were rejected and reverted.

An additional 2026-10-09 experiment at `61269c4` swapped the existing Laya
`matrixmultiply` GEMM operands and strides to compute `Yᵀ = W Xᵀ`, retaining
the same output layout and column-task partition. Existing scalar-oracle and
parallel GEMM tests passed, but full-model cached tenferro latency did not
improve overall: L8 B1 100.8 → 104.7 ms, L16 B1 108.9 → 108.5 ms,
L64 B1 192.7 → 197.8 ms, and L8 B8 186.0 → 184.0 ms. These were sequential
same-machine runs, portable provider, eight Rayon threads, two warmups and
median of five measured forwards. The small batch improvement does not justify
the regressions/variation elsewhere; the source change was rejected and restored.
Reports are in
[`bench-laya-gemm-orientation-2026-10-09`](../../../../fixtures/bench-laya-gemm-orientation-2026-10-09).

## Optional system OpenBLAS projections (2026-10-08)

`cpu-kernels/openblas` links an installed **LP64** OpenBLAS (`libopenblas`)
for the existing row-major `input_mul_weight_transpose_add_into` path when
there are at least 32 rows and dimensions fit CBLAS integers. Smaller inputs
keep the portable Rayon/matrixmultiply implementation. This is reached by
Laya's tenferro GEMM/bias extension; it does not replace tenferro dispatch.
All other CPU kernels retain their existing implementation. This feature is
opt-in and requires the system library at build/run time; the default build
has no BLAS linkage. Configure BLAS threads explicitly, for example:

```sh
OPENBLAS_NUM_THREADS=8 RAYON_NUM_THREADS=8 cargo run --release -p laya-infer --features cpu-kernels/openblas --example bench_laya_tenferro_gap -- "$LAYA_CHECKPOINT" 2 5
```

Same Ryzen/Linux setup as `21_SPEED_COMPARISON.md`, float32, eight Rayon
threads, sequential runs, warmup 2 / median of 5. System OpenBLAS 0.3.26:

| production cached Laya | portable baseline | OpenBLAS 8 | OpenBLAS 8 repeat | OpenBLAS 1 |
|---|---:|---:|---:|---:|
| L8 B1 | 105.3 ms | 103.0 ms | 105.9 ms | 103.5 ms |
| L16 B1 | 116.9 ms | 114.3 ms | 124.8 ms | 114.4 ms |
| L64 B1 | 225.0 ms | 216.9 ms | 215.1 ms | 618.4 ms |
| L8 B8 | 216.6 ms | 201.6 ms | 201.7 ms | 599.3 ms |

The 8-thread results improve L64/L8B8 by approximately 4%/7%; the 1-thread
configuration is substantially slower. The row threshold excludes L8/L16 B1,
so differences there are run variation, not provider speedups. The initial
experiment replaced the column-major Jeff kernel instead; those runs did not
test Laya's primary projection route and are excluded from this table.
Sanitized benchmark reports, environment metadata and the exact temporary
provider patch are in `fixtures/bench-openblas-cpu-2026-10-08/`. That patch
predates the final feature guard and layout/accumulation regression test.
The existing CPU and production checkpoint parity tests remain the correctness
gates; these timing reports alone do not prove numerical parity or resolve
issue #2's remaining gap versus Julia/Python.


## Tiled library GEMM for Laya attention (2026-10-09)

The existing CPU attention extension now uses `matrixmultiply::sgemm` for
QKᵀ and probability/value contractions at sequence lengths of at least 32.
Queries are tiled in groups of 64. A per-batch mask plan bounds the key span
for each tile and is shared across heads. GEMM is selected only when each
mask tile has at least half of its query/key rectangle allowed; very sparse
masks retain the scalar contractions. Nonfinite values also retain the scalar
path, preserving its behavior for masked NaN/Inf values. All-masked rows still
produce NaN. Checked storage sizes protect subsequent raw-pointer accesses.
The scratch score buffer is bounded by 64 times the largest key span.

This extends the existing CPU tenferro attention op with library GEMM. Native
model routing and other backends retain their existing operations. It does not
establish GPU performance or correctness.

Actual checkpoint timings used eight Rayon threads, two warmups per mode and
15 alternating pairs in one process with shared prepared weights. Benchmarks
ran sequentially. The temporary selector was removed from production code.

| L64 B1 | scalar attention | tiled GEMM attention | reduction |
|---|---:|---:|---:|
| first run | 194.394 ms | 178.215 ms | 8.3% |
| repeat | 191.375 ms | 179.009 ms | 6.5% |

L8/L16 B1 and L8 B8 retain the same attention algorithm; timing differences
there are variation. Maximum checkpoint logit/action absolute difference was
0.00048828125, within the production comparison tolerance of
`2e-3 + abs(reference) * 2e-5`. The production Julia question/action reference
test passed with the actual checkpoint (11.36 seconds).

Boundary tests cover lengths 31/32/33/63/64/65/127/129, full and sliding
attention, padding, mask holes and very sparse masks. Kernel tests additionally
cover masked nonfinite values, empty rows and invalid input storage.
Microbenchmarks show about 3.2x faster full attention at L64 and 5.7x at L512;
very sparse masks pay a small planning overhead, so these kernel gains must
not be interpreted as whole-model gains. Raw samples, replay patch against
`e1f98c4`, microbenchmark source and metadata are in
[`bench-laya-attention-gemm-2026-10-09`](../../../../fixtures/bench-laya-attention-gemm-2026-10-09).

## Rejected prepared weight layout (2026-10-09)

A separate experiment cached projection weights in a GEMM-friendly transposed
layout. Synthetic contractions improved by 10–43%, but two alternating-pair
checkpoint runs showed only small or mixed warm improvements. The measured
first forward grew from 2.01 to 2.97 seconds, including cache preparation and
computation, with checkpoint loading excluded. Outputs were identical.
The production prototype was restored rather than adopting that startup cost
for inconclusive warm gains. Exact prototype and measurement artifacts are in
[`bench-prepared-weight-layout-2026-10-09`](../../../../fixtures/bench-prepared-weight-layout-2026-10-09).


## Fused projection and GeGLU (2026-10-09)

`tenferro-ext::GemmGegluOp` combines the MLP up projection, optional bias
and exact GeGLU in one eager CPU operation. It calls the existing library
GEMM and `cpu-kernels` activation, so the arithmetic is unchanged. Activations
keep their feature-first trailing axes without separate flatten/restore eager
operations. The expanded projection stays in one runtime-local scratch buffer,
with retained capacity reported to the extension cache. Each returned tensor
owns its output storage; later calls can resize and overwrite scratch safely.
Laya selects this operation at 16 or more trailing columns. Shorter inputs
retain the original GEMM/GeGLU sequence; other backends retain native composition.

Production checkpoint, F32, eight Rayon threads, five warmups per mode,
15 alternating pairs, sequential benchmark processes:

| L64 B1 | separate ops | fused ops |
|---|---:|---:|
| initial pair run | 186.823 ms | 178.587 ms |
| repeat | 186.063 ms | 179.501 ms |
| cache + short-input threshold | 180.471 ms | 178.518 ms |

All paired logit/action differences were zero. Short inputs use identical
operations in the last run; their timing differences are variation. The final
cache/threshold result is smaller than the initial 3.5–4.4% improvement, so do
not attribute the initial gain to scratch caching. Tests compare separate and
fused operations bit-for-bit across strides, batches, optional bias and irregular
sizes, reject dtype/shape errors, and retain old outputs across scratch growth
and changed inputs. The final cached implementation passed the actual Julia
production question/action reference and the full workspace tests. Workspace
formatting, all-target Clippy with warnings denied, and CUDA-feature Clippy
also passed. Raw samples and a replay patch against `8ca8b81` are in
[`bench-fused-mlp-2026-10-09`](../../../../fixtures/bench-fused-mlp-2026-10-09).

A separate diagnostic replaced projection GEMM with the MKL library bundled
with PyTorch 2.14.1 (MKL 2024.2). It made production Laya slower: L8 approximately
99.5→118.6 ms and L64 185.6→194.9 ms. It was rejected and removed. Synthetic
MKL packed-weight tests showed mixed gains; they do not establish a model-level
speedup. The library remains a diagnostic dependency only. Source patches,
synthetic checks and fresh Python/native baseline records are in
[`bench-python-goal-blas-2026-10-09`](../../../../fixtures/bench-python-goal-blas-2026-10-09).

## Blocked library-packed weights diagnostic (2026-10-09)

An MKL diagnostic cached packed weight blocks, used eight Rayon workers with
one MKL thread per worker, and fixed 32-token tiles. It left the unpacked host
reference unchanged for output checks. One full-model run (F32, five warmups,
15 samples) measured L8 B1 99.694 ms, L16 B1 107.390 ms, L64 B1 162.436 ms
and L8 B8 159.583 ms. Maximum logit/action error was 0.001220703125 and passed
the model comparison tolerance. This still misses Python's approximately
134.5 ms L64 result.

The process reached 19034120 KiB RSS: the library's packed allocation sizes
for small output blocks greatly exceed the raw weights. The provider was
removed rather than adopted with this memory cost; these measurements are a
single diagnostic run, not a repeated production speedup. Microbenchmarks
also validate token tails at lengths 1/3/7/8/9/16/64/65. Exact source, replay
patch, raw samples and configuration are in
[`bench-mkl-blocked-2026-10-09`](../../../../fixtures/bench-mkl-blocked-2026-10-09).
