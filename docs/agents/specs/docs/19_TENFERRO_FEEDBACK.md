# tenferro-rs Feedback (filed as issues)

Issues filed upstream on `tensor4all/tenferro-rs` while building this stack
against revision `471c4278` (workspace version `0.7.1`).

## Release / feature gating

| # | Issue | Severity |
|---|---|---|
| [#1971](https://github.com/tensor4all/tenferro-rs/issues/1971) | crates.io `0.7.1` and git `471c4278` expose different `with_backend_session` signatures under the same version | high |
| [#1972](https://github.com/tensor4all/tenferro-rs/issues/1972) | `EagerSessionLinalgExt` is gated behind the `autodiff` feature, blocking inference-only linalg use | high |

## Missing inference ops

| # | Issue |
|---|---|
| [#1973](https://github.com/tensor4all/tenferro-rs/issues/1973) | Add an elementwise `erf` op (needed for exact GELU) — **closed as not planned**; handled by a self-hosted extension op |
| [#1974](https://github.com/tensor4all/tenferro-rs/issues/1974) | Add `cumsum` / `cumprod` |
| [#1975](https://github.com/tensor4all/tenferro-rs/issues/1975) | Add `sigmoid` / `silu` / `softplus` / `gelu` convenience ops |
| [#1976](https://github.com/tensor4all/tenferro-rs/issues/1976) | Add `argmax` / `reduce_mean` / `softmax` / `log_softmax` |
| [#1977](https://github.com/tensor4all/tenferro-rs/issues/1977) | Add a depthwise causal `conv1d` op |
| [#2006](https://github.com/tensor4all/tenferro-rs/issues/2006) | Add fused CPU `layer_norm` / `rms_norm` ops (the eager form is ~10 composed ops) |

## dtype / backend coverage

| # | Issue |
|---|---|
| [#1978](https://github.com/tensor4all/tenferro-rs/issues/1978) | Add public `F16` / `BF16` (and `I8`) dtypes for inference |
| [#1979](https://github.com/tensor4all/tenferro-rs/issues/1979) | WebGPU/Metal coverage is effectively `dot_general` only; document or broaden |

## API ergonomics

| # | Issue |
|---|---|
| [#1980](https://github.com/tensor4all/tenferro-rs/issues/1980) | `dot_general` batch-trailing output makes attention/GQA layout error-prone |
| [#1981](https://github.com/tensor4all/tenferro-rs/issues/1981) | `Tensor` does not implement `Clone` |
| [#1982](https://github.com/tensor4all/tenferro-rs/issues/1982) | `with_eager_session` requires `Send` and returns a nested `Result` (`??`) |
| [#1983](https://github.com/tensor4all/tenferro-rs/issues/1983) | Provide public error constructors on `tenferro_ad::Error` |
| [#1984](https://github.com/tensor4all/tenferro-rs/issues/1984) | `from_vec_col_major` silently reinterprets row-major data |
| [#1985](https://github.com/tensor4all/tenferro-rs/issues/1985) | `reduce_sum_squares` axes argument differs from other reductions |
| [#1986](https://github.com/tensor4all/tenferro-rs/issues/1986) | `constant_from` vs `constant_from_host` naming is ambiguous |
| [#2005](https://github.com/tensor4all/tenferro-rs/issues/2005) | Extension ops reach `dot_general` via `BackendSession` but not linalg/composed primitives, so a tensor-native fused op cannot solve (revises the note in `23_TENFERRO_NATIVE.md`) |

## Performance

| # | Issue | Severity |
|---|---|---|
| [#1990](https://github.com/tensor4all/tenferro-rs/issues/1990) | CPU fused elementwise region is ~3.7× slower than eager per-op kernels for a long elementwise chain | medium |
| [#1992](https://github.com/tensor4all/tenferro-rs/issues/1992) | CPU eager `dot_general`: GEMM analysis recomputed per call (no plan cache) and the BLAS provider cannot execute linalg | medium |
| [#1995](https://github.com/tensor4all/tenferro-rs/issues/1995) | CPU eager decode forward: small-`m` GEMMs (faer) and internal layout copies (`structural::typed_copy_into_uninit`) dominate; GEMM providers are shape-dependent | medium |
| [#2003](https://github.com/tensor4all/tenferro-rs/issues/2003) | CPU eager `dot_general` is ~2.3× a plain host GEMM at decode (per-call overhead 1.7× over raw faer + faer-vs-`sgemm` kernel choice); no eager provider selection | medium |
| [#2007](https://github.com/tensor4all/tenferro-rs/issues/2007) | `triangular_solve` / `solve` are rank-2 only (no batch dims), forcing per-batch eager loops | medium |

MWE and benchmark for #1990:

MWE: `crates/bench-suite/examples/fused_elementwise_mwe.rs`

```sh
RAYON_NUM_THREADS=8 cargo run --release -p bench-suite --example fused_elementwise_mwe -- 200 20
```

A traced chain of `K` `tanh` ops over a `1024x64` `f32` tensor compiles to a
**single elementwise region** that is executed as **one fused command**:
`PreparedCompiledGraph::elementwise_region_summary()` reports `regions=1,
covered_insts=K`, and `_execution_counts()` reports `fused=runs, fallback=0`.
Yet the compiled time scales linearly and is ~3.7× the eager per-op path:

| `K` | eager (per-op kernels) | compiled (1 fused region) | compiled / eager |
|---:|---:|---:|---:|
| 1 | 0.23 ms | 0.27 ms | 1.16× (no region) |
| 10 | 1.79 ms | 5.96 ms | 3.34× |
| 100 | 17.3 ms | 63.9 ms | 3.69× |
| 200 | 34.7 ms | 128.0 ms | 3.69× |

Fusion is entered and run (zero fallbacks), so this is a **fused-kernel
quality** issue (`tenferro-cpu-fused` / `strided_fused` on CPU), not a
dispatch/segmentation issue: the fused region's per-element cost exceeds the
eager path's sequential per-op SIMD kernels.

## Raw CUDA ownership caveat (2026-10-08)

At the pinned revision, `cuda::raw::Module` and `Function` are `!Send`/`!Sync`.
`raw::Session::resource` requires `T: Send`; the runtime's
`ExtensionCacheStore::put` requires `Send + Sync`. Loaded module handles
therefore cannot be cached directly in either store. A thread-bound CUDA
execution owner must retain the module instead; do not add unsafe `Send`/`Sync`
implementations to bypass this boundary. `BackendSessionHost::with_backend_session`
also requires both its closure and return value to be `Send`, so a loaded
module cannot be captured from outside or returned across session admission.
Construct/use a thread-bound owner inside the admitted executor scope, or
provide an appropriate thread-local cache; runtime identity must be checked.

`raw::Session::launch` explicitly requires modules and device allocations to
remain live until a subsequent synchronization; launch does not retain them
for asynchronous completion. An extension-local module that is dropped on
return is insufficient. Full GatedDelta integration must provide an execution
owner/lease that spans the device work, with stream-ordered buffer reuse and
cleanup. The current low-level `CudaKernels` owner validates and enqueues raw stages
through an unsafe API; it leaves completion/lifetime management to the caller.
The ignored CUDA stage test explicitly synchronizes and leaks retained GPU
resources on synchronization failure. This test is not the full inference
execution owner.

## Model extension routing (2026-10-09)

F32 dtype alone does not imply CPU residency. Laya and Jeff now select the
CPU-only linear/GEMM, normalization, attention and gated-activation extensions
only when the admitted backend exposes the public CPU execution marker
(`with_cpu_exec_session`). Other backends use the existing native session
compositions. Jeff's dense linear also uses native `dot_general` for other
dtypes, with the same `(in, out)` weight layout. No host transfer or backend
switch is used to satisfy an unsupported op.

Laya's erf-based GELU now has a tensor-native composition in `tenferro-ext`:
F32 uses the CPU MLX polynomial coefficients with native `expm1`, and F64
uses the same Abramowitz–Stegun approximation as the CPU extension. CPU
sessions retain the original fused extension. Other sessions select the native
composition and upload only scalar constants. No tanh GELU substitution is
used. Native F32 arithmetic is not bit-identical to the fused CPU polynomial;
CPU verification over 40,001 grid points plus branch boundaries measured erf
maximum absolute difference 8.34465e-7; a separate 40,001-point GELU grid
measured 5.9604645e-7. Special-value tests cover signed zeros, infinities,
extreme finite F32 values and NaN. CUDA execution has an explicit
ignored hardware gate (`tenferro-ext/cuda`); actual device parity remains
unverified on this machine. Unsupported native primitives still return errors.

This does not establish full GPU model support. Jeff's default HostRecurrent
DeltaKernel remains explicitly CPU-only; TensorNative is the portable
formulation. Raw CUDA request integration, full-model GPU parity and backend
primitive coverage remain outstanding work. The native erf convenience path
uploads scalar coefficients per call; cached erf/GELU entry points now reuse
floating scalar tensors through `TensorCache`. Scalar keys use effective F32
or F64 bits (preserving signed zeros), and scalar entries are cleared when the
input runtime changes. Laya's cached forward uses the cached native GELU at
all three activation sites on non-CPU sessions. CPU fused activation dispatch
is retained. This covers these activation coefficients, not every model
constant or input on every backend.

At the pinned revision, eager `scale_real` constructs a host tensor then
calls `constant_from`, whose CUDA materialization expects a device-resident
view. Native erf/GELU therefore uses explicitly uploaded scalars and `mul`;
the cached variant reuses those tensors. Shared inference norm/attention/tanh
GELU scaling also uses explicit upload plus `mul`. Native DeltaNet prepares
its inverse key-width square-root scalar with its reusable constants and uses
`neg` for softplus's negative absolute value. No pinned dependency was changed.
This avoids the known implicit scalar-placement boundary in these eager model
paths; actual GPU behavior remains subject to hardware validation. Traced
`scale_real` behavior and other dtype/primitive coverage are separate concerns.

A diagnostic release run forced only Laya's GeGLU/erf/GELU compositions onto
the native path on CPU and passed the actual production
`bundled_questions_match_julia_collate_predict_and_action` test (Julia logits,
action scores and answers). The temporary routing overrides were restored;
other model operations retained their CPU paths. This provides full-model CPU
accuracy evidence for the new activation, not GPU evidence. The conditions,
existing tolerance and result are recorded in
[`native-erf-production-2026-10-09`](../../../../fixtures/native-erf-production-2026-10-09/report.json).

A follow-up diagnostic selected the cached native GELU at every Laya activation
site and passed the same production Julia logits/action/answer test. The
conditions and result are in
[`report-cached.json`](../../../../fixtures/native-erf-production-2026-10-09/report-cached.json).
Cache regression tests also exercise changed inputs, strided views, old-output
retention, scalar bit/dtype identity and runtime replacement. The CUDA gate
now retains plain and cached erf/GELU outputs across three changed-input
requests before downloading and comparing them; it remains unverified on
hardware because libcuda is absent.

## Positive findings (not filed)

- `triangular_solve(..., unit_diagonal = true)` fits the chunked Gated DeltaNet
  effective system exactly (the operator is applied and the diagonal ignored).
- `dot_general` batch dimensions are sufficient for attention / GQA.
- The eager session surface covers the inference primitives (norm, activations,
  softmax, RoPE, attention).
- Unsupported dtype/shape returns a typed error with no silent CPU fallback.


## Prepared CPU library projection workaround (2026-10-09)

The pinned eager path still lacks an exposed owned packed-weight CPU projection
resource reusable across repeated model forwards. Recreating layouts and
projection buffers is material at Laya's production sizes. The optional
self-hosted `PreparedGemm` extension uses standard oneDNN primitives, owns packed
weights independently of the runtime, and stores scratch in session caches.
The implementation and lifetime/thread tests are in this repository; no pinned
tenferro source or submodule is modified. Non-CPU backends retain the native
composition. A future public tenferro preparation interface could replace this
CPU-only provider while preserving model-level caching and independent outputs.


## CUDA engine findings (2026-10-10)

Hardware validation of the CUDA engines (pinned `471c427`, RTX 3060) found:

- **Bool broadcast**: `broadcast_in_dim` + `where_select` on a Bool mask fails
  in `CudaBackend::to_contiguous_read` (`UnsupportedDType: Bool`). Device
  attention now adds an F32 score bias instead.
- **Eager `scale_real`** still imports a host scalar that CUDA cannot read
  (the cause of two failing `cuda_stages` gates); tests now upload the scalar.
- **Mixed dtypes**: an F64 state silently promoted the F32 chunked scan to an
  F64 result on CUDA (CPU would reject it); the test literal was fixed.
- **`gather` over a `(vocab, hidden)` table** permutes the entire table per
  call (6.3 ms, 1 GB for Jeff); even a `(hidden, vocab)` table did. A
  first-class embedding/gather-rows op without the permute would remove the
  need for the fused kernel.
- **`dot_general` layout**: contracting a weight's first axis produced an
  output permute (`cutensor permute_coop`), contracting its second axis did
  not.
- **Raw seam costs**: every `CudaExecSession::with_raw` call flushes CubeCL
  with a `cuEventSynchronize` (measured with an empty closure), and
  `raw::Session::tensor` only binds owned `TypedTensor`s. Eager constants
  (`constant_from[_host]`) are pooled views, so they must be copied once
  before raw use; eager op outputs are owned and bind in place. A
  non-blocking raw session and a view-binding API would let fused kernels
  avoid both.
- **`reshape` of a constant** materializes a copy on CUDA; rank-1 weights are
  now cached directly.
- **Per-op host overhead** (~40 µs per eager op incl. cuBLAS/cuTENSOR plan
  lookup and allocation) bounds small-batch latency once kernels are fused.
