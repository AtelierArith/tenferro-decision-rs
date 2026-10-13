# Tenferro-Native Execution

**Status:** Eager forwards and caching implemented; full device-resident model execution remains incomplete

**Date:** 2026-10-09

**See:** `AGENTS.md`, `21_SPEED_COMPARISON.md`, `22_CPU_KERNEL_OPTIMIZATION.md`

## Why

Model computation should run through tenferro so every backend tenferro
supports (CPU today; CUDA / WebGPU / Apple GPU later) shares one code path. The
all-host `forward_reference` stays only as the correctness oracle.

The production path (`forward_tenferro`) was already tenferro-native
(embeddings, norms, attention, Gated DeltaNet, MLP, readout all use
`EagerSession` ops), but it was slow because it re-transposed and re-created
every weight tensor on every call.

## What changed

- **`tenferro-infer::TensorCache`** — a reusable column-major weight-tensor
  cache keyed by host storage identity. `EagerTensor` keeps an
  `Arc<EagerRuntime>`, so cached weights are usable by any later session that
  shares the runtime. Two constructors: `col` (row-major host data, transposed
  once) and `col_major` (already column-major data, e.g. Jeff's embedding table
  where row-major `(hidden, vocab)` *is* column-major `(vocab, hidden)`).
- **`laya-infer`** — `forward_tenferro_cached` (and
  `forward_encoder_tenferro_cached`) thread the cache through; `LayaEngine` owns
  an `Arc<EagerRuntime>` and a `TensorCache` and runs the tenferro forward.
- **`jeff-infer`** — `forward_tenferro_cached` threads the cache plus the
  DeltaNet workspace; `JeffEngine` owns the runtime, workspace, and cache.
- **`tenferro-gated-delta::tensor_layer`** — a fully tensor-native,
  **head-batched** Gated DeltaNet (mask, causal depthwise conv, Q/K L2-norm,
  gates, chunked scan, output projection). `DeltaKernel::TensorNative` keeps the
  DeltaNet entirely in-session for backend portability.
- **Host recurrent DeltaNet via the `GatedDelta` extension op** — the default
  `DeltaKernel::HostRecurrent` runs the fused host recurrent kernel *through the
  session* (`EagerSessionGatedDeltaExt::gated_delta`). The op reads every weight
  in place (the column-major buffers *are* the kernel's row-major operands) and
  borrows them (`GatedDeltaWeightSlices`), so switching kernels costs no extra
  copies. This keeps the forward tenferro-first while calling the fastest CPU
  kernel on the DeltaNet. On a non-CPU backend this explicit selection reports
  an unsupported extension; callers must select `TensorNative` themselves.
- **`(length, hidden)` end-to-end** — the Jeff tenferro forward keeps hidden
  states in `(length, hidden)` (embedding gather, rms_norm over the last axis,
  `linear` contracting the last axis), so there are no per-layer
  `(hidden, length) ↔ (length, hidden)` transposes and the `GatedDelta` op's
  `(length, hidden)` output is zero-copy. The measured wall-clock effect is
  within noise, but it removes redundant ops (see `22_CPU_KERNEL_OPTIMIZATION.md`).
- The engines' `decide` / `logits` / `row_logits` now take `&mut self` for the
  cache (the eager session API requires a `Send` closure, so a `RefCell` borrow
  cannot cross it).

CPU fused model extensions are selected using the admitted CPU execution
marker as well as F32 dtype. Other backends use native dense, norm, attention
and gated-activation compositions. This routing does not yet establish full
GPU support: Laya's erf-based GELU now has a native polynomial composition,
with scalar coefficients reused by the model cache. Eager native scaling uses
explicitly uploaded scalars, and DeltaNet retains its Q multiplier with its
prepared constants. Device parity and CUDA request integration remain pending (see
`19_TENFERRO_FEEDBACK.md`).

### tenferro conventions pinned down

- `dot_general` emits **batch dimensions trailing**; `tensor_layer::bmm`
  transposes them back to batch-leading.
- `reshape` preserves **column-major** order, so splitting `[width, L]` into
  heads reshapes to `(dim, heads, L)` then transposes (as attention already
  does).
- `triangular_solve` is rank-2 only, so the per-chunk solves loop over heads
  while everything else stays batched.

## Measured (production checkpoints, release, 8 threads)

Laya, L8B1:

| path | median |
|---|---|
| tenferro, fresh cache each call | 1370 ms |
| **tenferro, reused cache** | **160 ms** (8.6×) |
| host + GEMM + rayon (oracle) | 216 ms |

The cached tenferro path is the fastest Rust Laya forward (within ~1.7× of
Julia's OpenBLAS).

Jeff, L8 (host `+GEMM+rayon` is the oracle / CPU best):

| path | L8 | L64 |
|---|---:|---:|
| host + GEMM + rayon | 169 ms | 739 ms |
| tenferro, fresh cache each call | 4677 ms | — |
| tenferro, reused cache, old host-round-trip chunked path | 1693 ms | — |
| tenferro, reused cache, tensor-native (head-sequential) | 496 ms | — |
| tenferro, reused cache, tensor-native, head-batched | 344 ms | 894 ms |
| **tenferro, reused cache, host recurrent DeltaNet (`GatedDelta` op)** | **292 ms** | **629 ms** |

Caching plus the tensor-native rewrite cut the tenferro path ~5× (1693 → 342 ms).
Routing the DeltaNet through the fused host recurrent kernel (`DeltaKernel::HostRecurrent`,
the default, via the zero-copy `GatedDelta` extension op — see below) cuts it
further, to ~1.7× the host at L8 and **0.71× at L64** (the tenferro path is
faster there). The remaining L8 gap is the tenferro eager-op / small-GEMM
overhead spread across attention/MLP; the payoff is a single backend-portable
code path that reaches the fastest CPU kernel per op.

## Backend selection

Because the two paths differ by model, the engines let the caller choose:

- `LayaEngine` runs the tenferro forward by default (it is faster).
- `JeffEngine` defaults to `JeffBackend::Host` and offers
  `JeffBackend::Tenferro` (via `with_backend`) for backend portability / future
  GPU execution.

The host forwards remain the correctness oracle.

## `GatedDelta` extension op

Making the extension op execute through tenferro session ops is **not feasible**
with the current extension API, but for a narrower reason than first recorded:
`execute_in_session` receives `&mut dyn BackendSession`, which *does* expose the
raw tensor ops (`TensorBackendOps`, including `dot_general`, elementwise,
structural, reduction). What is out of reach is **linalg** — `triangular_solve`
/ `solve` live on `EagerSessionLinalgExt` (an eager-session trait), not on
`BackendSession` — and the composed `tenferro-infer` primitives (norm /
attention / RoPE). The chunked scan needs a solve, so the op keeps its CPU fused
kernel (tracked as tensor4all/tenferro-rs#2005).

Because the host recurrent kernel is faster than the tensor-native chunked scan
on CPU (measured ~1.8× on the layer; see `bench_delta_paths`), the Jeff tenferro
forward **uses the extension op by default**: `DeltaKernel::HostRecurrent`
invokes it through the eager session, so the forward stays tenferro-first and
the kernel is the host fast path. `DeltaKernel::TensorNative` selects the
in-session chunked formulation instead (what a future non-CPU backend would
need), and is available via `JeffEngine::with_delta_kernel`. The forward is
tenferro-native on every other op regardless.

The op is zero-copy: the inputs are the host kernel's operands in row-major
order held as column-major tenferro tensors (`prepare_kernel_weights` builds
them with `TensorCache::col_major`, no transpose), and `delta_layer_recurrent_slices`
borrows them via `GatedDeltaWeightSlices` instead of owning copies. Feeding the
old "textbook" column-major `(in, out)` layout made the op transpose every
weight per call and run 3–11× slower than the bare kernel; the row-major input
layout removes that entirely.

## Host-kernel extension ops (Laya, Jeff)

After caching, the remaining gap was the eager per-op overhead of the many
small ops. The hot chains now run through self-hosted `tenferro-ext` ops backed
by `cpu-kernels`:

- **Laya** — `linear` / `gemm_bias` (feature-first `y = Wᵀx`; the `(in, L, B)`
  → `(in, L*B)` reshape is layout-preserving, so no transposes), a fused
  feature-first LayerNorm (replaces transpose → norm → transpose), GeGLU
  (`gelu(value)·gate`), and one fused **split + RoPE + masked attention**
  block. The block runs entirely in the host-friendly feature-first
  `(d, L, B)` layout (per-head `head_dim` contiguous), so the head transposes
  are never materialized.
- **Jeff** — `linear` (`x (length, in) · W (in, out)`), fed the raw row-major
  `(in, out)` weight and column-major activations, the orientation
  `matrixmultiply` vectorizes (`rsa = 1`, `csb = 1`); the full-attention block
  (centered RMSNorm ×2 + partial RoPE ×2 + causal masked attention + sigmoid
  gate); the feature-last RMSNorm; and gated SiLU. Caching linear weights with
  `TensorCache::col_major` over the raw buffer gives the op that storage.

With a reused `TensorCache` these put the cached tenferro forward at the Rust
host path for Laya and ~1.1–1.2× `host_opt` for Jeff
(`21_SPEED_COMPARISON.md`). A standalone masked-attention op and a standalone
`split_qkv` op were tried and reverted: at the small decode shapes their
extension-op fixed cost cancelled the saved eager ops.

## Next

- Apply the same prepared-tensor approach to any remaining per-call host work
  (e.g. attention mask construction) and to future backends.
- When tenferro exposes device backends, weight tensors prepared the same way
  can be kept device-resident.


## Explicit output transfer (2026-10-09)

`tenferro-infer::output::host_value` materializes an eager output in logical
column-major order, validates session/runtime ownership through
`duplicate_value`, then calls the admitted backend's `download_to_host` for
backend storage. Host tensors retain the existing single materialization.
Laya's `extract_col` and Jeff's final readout use this boundary. Previously,
these helpers assumed `duplicate_value` produced host storage, although on a
GPU it only copies within the device and `as_slice` cannot read those bytes.

CPU tests cover transposed views, retained outputs, repeated readout and foreign
runtime rejection. The ignored CUDA gate
`cuda_output_boundary_downloads_strided_retained_outputs` covers the same
logical layout across three requests and repeated downloads. This gate needs
CUDA hardware; CPU tests and CUDA compilation do not prove device execution.

This fixes the transfer mechanism, not the remaining model integration. Laya
still computes marker pooling on the host and transfers that intermediate
back for its action head. The pinned revision lacks CUDA Bool leaf materialization; the numeric-mask
construction described below avoids importing a Bool leaf. Jeff's explicitly
selected host recurrent path has intermediate host reads; use the native path
for device computation. GPU engine admission and raw CUDA request integration
remain separate work. Full GPU model parity and latency remain unverified.

Validation: workspace tests and both ordinary/CUDA-feature workspace Clippy
passed. The actual Laya Julia question/action reference test passed in release
(11.01 seconds). Explicitly running the CUDA output gate failed before any
tensor computation because `libcuda` is absent; device parity is unverified.


## Native Bool mask construction (2026-10-09)

`tenferro-infer::input::bool_tensor_native` uploads masks as F32 zero/one
values and obtains a backend Bool tensor with `compare(..., zero, Gt)`.
This avoids the pinned CUDA eager Bool-leaf materialization gap. Laya and
Jeff attention use it on non-CPU sessions; CPU sessions retain direct Bool
imports, avoiding additional conversion/allocation work in current inference.
No failed Bool import is caught or silently retried on the host.

CPU tests cover mixed/all-true/all-false column-major masks, selection,
transposed retained outputs, multiple requests, scalar masks and shape errors.
The ignored CUDA regression `cuda_native_masks_select_without_bool_leaf_import`
checks device comparison and selection before downloading the resulting F32
outputs. This is implementation and CPU evidence; GPU execution still requires
hardware validation and does not establish complete GPU model integration.

Validation: workspace tests (including the actual Laya production reference)
and ordinary/CUDA-feature workspace Clippy passed. The explicit native-mask
CUDA gate failed before computation because `libcuda` is absent.


## Owned optional oneDNN CPU extensions (2026-10-09)

The `laya-infer/onednn` feature routes CPU F32 projections through a prepared
extension op. Its model-owned packed weights do not retain an `EagerRuntime`;
cloned tensor caches share preparation without creating a runtime cycle.
Session cache entries own reusable projected activations and aligned native
scratch. The native provider uses standard oneDNN inner product, erf GELU,
binary multiplication and LayerNorm primitives with strict F32 math; it contains
no bespoke GEMM or SIMD arithmetic. Non-CPU sessions retain tenferro composition.

Native primitives use user-provided scratch and complete execution before
clearing borrowed data handles. This is required for sequential execution on a
thread different from the preparation thread; library-managed scratch cannot
supply that guarantee in every oneDNN build (see the
[oneDNN scratchpad contract](https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_scratchpad.html)).
A mutable prepared primitive is serialized by its owner. Projection and gated
plan caches are bounded to 32 shapes. LayerNorm is selected only for width 1024,
at least 16 columns, and a positive finite epsilon; other shapes use the existing
CPU extension. Its shape-plan cache is also bounded to 32 entries.

Laya attention retains per-head Q/K/V, probability and mixed-value buffers in
session caches separated by RoPE base. Input and mask data are read afresh on
every call. Tests exercise growing and shrinking shapes, changing heads, masks,
RoPE settings and inputs. Final output tensors own independent storage.

The owned-provider benchmark fixture retains raw samples and upstream Python
outputs. Two attention-workspace runs beat the measured upstream Python at L8
B1, L64 B1 and L8 B8; the L64 margin is small. The actual bundled Julia question
and action reference passes with these changes. Workspace formatting, all-target Clippy and tests passed; oneDNN-enabled
workspace Clippy and related release tests passed. Jeff also beats fresh
upstream Python across all three measured shapes in two Rust runs. No device
performance claim follows from these CPU measurements. Build instructions are
in the README.


## CUDA execution (2026-10-10)

`Device::Cuda` runs the same `forward_tenferro` code on tenferro-gpu
(RTX 3060, CUDA 13 devcontainer). Weights are prepared once per runtime in
the engine's `TensorCache` (plus per-thread fused-kernel copies) and stay on
the device; a call uploads ids/masks and downloads logits.

### What dominated, and what changed

Profiles (`nsys`, Jeff parcel L101) drove the work:

| step | events / forward | GPU busy | warm median (quiet host) |
|---|---:|---:|---:|
| first working device path (composed eager ops) | 3284 | 66 ms | 130 ms |
| + fused norm / SiLU / attention kernels | 1056 → 934 | 41 ms | 48 ms |
| + embedding gather fix, `(out, in)` DeltaNet weights | 812 | 36 ms | 39 ms |
| + tiled recurrent scan, stacked q/k/v/g, gate/up, qkv/z/a/b GEMMs | 728 | 34 ms | 38 ms |

- Composed RMSNorm/LayerNorm/GELU/attention were ~25–60 tiny launches each
  (plus scalar uploads); one fused NVRTC kernel replaces each chain
  (`tenferro_ext::cuda_fused`, selected through `Fusion`).
- Native `gather` permuted the whole embedding table (1 GB for Jeff, 6.3 ms)
  every forward; a fused gather reads the original host layout.
- `dot_general` with `(in, out)` weights produced an extra output permute;
  `(out, in)` weights map to a transpose flag of the same GEMM.
- The original recurrent scan reloaded and renormalized Q/K uncoalesced in
  every warp (1.6 ms/layer at L101); tiling through shared memory gives
  0.47 ms/layer.

### Measured (RTX 3060, release, warm median, includes upload/readout/sync)

Host load average 18–29 from unrelated jobs during this run (min in
parentheses). A later run at load average 6–9 (10 warmup, 50 iterations)
measured: Jeff parcel trimmed 37.9 ms, full L256 79.7 ms, L8 27.0 ms, L64
28.7 ms; Laya L8B1 26.5 ms, L64B1 29.5 ms, L8B8 27.8 ms.

| workload | CUDA | CPU host_opt | CPU tenferro |
|---|---:|---:|---:|
| Jeff parcel B1/L256 (101 active, trimmed) | 48.3 ms (43.7) | 661 ms | 789 ms |
| Jeff parcel full L256 (padding kept) | 76.2 ms (74.9) | — | — |
| Jeff synthetic L8 | 44.2 ms (32.6) | 162 ms | 351 ms |
| Jeff synthetic L64 | 32.9 ms (31.5) | 470 ms | 637 ms |
| Laya L8B1 | 40.2 ms (30.7) | — | 146 ms |
| Laya L64B1 | 47.3 ms (43.8) | — | 365 ms |
| Laya L8B8 | 41.2 ms (31.7) | — | 373 ms |

(The CPU columns ran on the same loaded host and are slower than the quiet
numbers in `21_SPEED_COMPARISON.md`.) For reference, JeffClient.jl's
QwenDecisionCore CUDA path measures 47.5 ms full-sequence / 25.9 ms trimmed
on this GPU: the trimmed Rust path is ~1.5x slower, the full-sequence one
~1.6x.

### Remaining cost

At L8 GPU work is ~10 ms but the forward takes ~31 ms: the remaining time is
host-side tenferro eager dispatch (~700 launches/forward: cuBLAS/cuTENSOR
plans, output allocation, small activation copies) and one
`cuEventSynchronize` per raw-CUDA session (`with_raw` flushes CubeCL).
Further gains need either fewer eager ops (e.g. fused residual add + norm,
device marker pooling for Laya, persistent cuBLAS handles behind a fused
GEMM op) or tenferro-side changes recorded in `19_TENFERRO_FEEDBACK.md`.


## Raw single-stream CUDA forward (2026-10-11)

The tenferro-native device forward above is launch- and host-bound (a ~25 ms
floor per call). The raw path (`CudaPath::Raw`, the CUDA default of both
engines) instead runs the whole forward in **one** `CudaExecSession::with_raw`
scope, as a self-hosted extension; tenferro-rs is unchanged and the native
path stays selectable (`CudaPath::Native`, `TENFERRO_DECISION_CUDA_PATH=native`)
and is the fallback for unsupported models.

- `tenferro_ext::raw_exec`: the scope (`with_raw_exec`), a cuBLAS handle per
  thread and runtime bound to tenferro's captured stream (cudarc 0.19, as in
  tenferro-gpu), an NVRTC module cache, async uploads, one synchronizing
  download, and `DeviceBuffer`s (driver allocations on the primary context,
  freed on drop; see `19_TENFERRO_FEEDBACK.md` for why not CubeCL's pool).
  Weights upload once into a few aligned arenas; activations live in one
  scratch buffer grown to the largest request, so a warm call allocates
  nothing.
- Laya (`laya_infer::cuda_raw`, `cuda/laya_raw.cu`): every linear layer is a
  cuBLAS `N, N` GEMM on transposed weights with residual adds folded in
  (`beta = 1`); warp-per-column LayerNorm; MLX-exact GELU/GeGLU; a port of
  Laya.jl's fused FP32 flash attention (64-query blocks, 32-key tiles, RoPE
  on staging, online softmax, local-window and padding tiles skipped, no score
  matrix); marker gather/scorer/pooling and the action head on device, one
  download of logits + actions.
- Jeff (`jeff_infer::cuda_raw`, `cuda/jeff_raw.cu`,
  `tenferro_gated_delta::raw` + `cuda/delta_raw.cu`): packed `[qkv|z|a|b]`,
  `[q|gate|k|v]` (GQA key/value heads de-duplicated) and `[gate|up]`
  projections stored `(in, out)` for cuBLAS `T, N` (QwenDecisionCore's
  orientation, measured faster at L8/L256; `TENFERRO_DECISION_JEFF_GEMM=nn`
  switches), centered RMSNorm with the DeltaNet input mask fused, causal conv +
  SiLU, Q/K L2 norm, gates, the scan — warp-per-row recurrence below 32 tokens,
  QwenDecisionCore.jl's chunked WY form (prepare kernel, batched cuBLAS
  products, blocked triangular inverse, sequential chunk hand-off) from 32 on —
  RMSNorm·silu(z) gate, per-head norm + partial RoPE (f64 host tables),
  gated causal attention (8 queries per block sharing K/V reads), and a final
  full-attention layer computed for the last position only.

Accuracy (max |Δ|, hardware gates): Laya raw logits vs CPU ≤ 7.2e-6 (B1, padded
B5, padded L300 B2 with sliding-window tile skipping), vs Julia ≤ 2.9e-6;
actions relative ≤ 1e-6. Jeff raw: Julia L8 vs CPU 6.7e-6 / vs Julia 4.8e-6;
parcel vs CPU 7.2e-6 / vs PyTorch 1.05e-5 (scale 18.6); synthetic stacks vs the
host oracle ≤ 6e-6 for both scans at lengths 1–200 with mask holes.

Measurements (RTX 3060, GPU 0, F32, warm median of 50 after 10 warmups,
including upload/forward/download/sync; Rust raw and Julia alternated, mean of
two runs; `fixtures/bench-gpu-raw-2026-10-11`):

| row | before (native) | after (raw) | Julia CUDA |
|---|---:|---:|---:|
| Jeff parcel trimmed L101 | 37.9 | 24.7 | 25.8 |
| Jeff parcel full L256 | 79.1 | 45.7 | 47.2 |
| Jeff synthetic L8 | 26.5 | 9.7 | 12.0 |
| Jeff synthetic L64 | 28.0 | 15.6 | 16.8 |
| Laya L8 B1 | 26.2 | 7.2 | 8.0 |
| Laya L64 B1 | 28.2 | 11.0 | 11.8 |
| Laya L8 B8 | 27.0 | 11.0 | 11.9 |
| Laya L93 B1 | 30.0 | 15.8 | 16.8 |
| Laya L93 B10 | 155.2 | 107.7 | 111.8 |
| Laya L512 B1 | 267.6 | 63.6 | 67.4 |
| Laya L512 B10 | OOM | 542.5 | 582.3 |

Where the time goes now: the calls are GPU-bound (Jeff L8: 9.7 ms of kernels
per 9.7 ms call). Jeff L256 spends ~80% in F32 cuBLAS SGEMM; the NN/TN choice
alone moves L64 and L256 by ~5% in opposite directions, so per-shape GEMM
algorithm selection (cublasLt heuristics/autotuning) is the next lever, then a
bandwidth-optimal skinny GEMM for ≤ 8 tokens (L8 reads ~2 GB of weights at
~57% of peak bandwidth).


## CubeCL kernels for the raw path (2026-10-11, experimental)

Question: can the raw path's non-GEMM kernels be `#[cube]` (CubeCL) kernels,
portable to CubeCL's wgpu/Metal runtimes, with GEMMs on cuBLAS, without
losing speed? Selected with `TENFERRO_DECISION_CUDA_KERNELS=cubecl` (or
`LayaCudaRaw::with_kernels` / `JeffCudaRaw::with_kernels`,
`tenferro_ext::raw_exec::RawKernels`); NVRTC stays the default.

### Seam measurements (RTX 3060, GPU 0, pinned `471c427`)

MWE `target/tenferro-mwe/20-cubecl-seam` (standalone crate):

| measurement | result |
|---|---|
| empty `with_cubecl`, idle / empty `with_raw`, idle | 15.8 µs / 8.4 µs per scope |
| empty scope behind 67 ms of queued GEMMs | `with_cubecl` blocks on the **first** scope (66.8 ms; its exit flush waits); `with_raw` on the second (67.2 ms, tenferro-rs #2078) |
| scope that launches a slow `#[cube]` kernel | returns only when the kernel has finished (exit flush is a host barrier) |
| alternating `with_cubecl`(copy) ↔ `with_raw`(256³ cuBLAS) | 0.8–1.7 ms per pair |
| streams | CubeCL kernels, cuBLAS on the raw stream and eager GEMMs all run on one CUDA stream (nsys) |
| `#[cube]` launch with a cloned client inside one raw scope, then cuBLAS | 0/100 stale reads, but unordered by contract (launches go to CubeCL's device thread asynchronously); `ComputeClient::exclusive(\|\| ())` drains the queue first: ordered by construction, ~17–20 µs |
| binding our `cuMemAlloc` buffers | impossible (no adoption API); CubeCL binds only tenferro pool tensors |
| freed pool memory (#2081) | `client.memory_cleanup()` returns it to the driver (6.5 → 10.5 GiB free) |

So per-op `with_cubecl` scopes are unusable for inference (each blocks), and
the hybrid that works runs the whole forward in **one raw scope**: `#[cube]`
kernels launched on tenferro's client (cloned once from a short
`with_cubecl` scope that allocates the pool tensors), cuBLAS and the
remaining NVRTC kernels on the raw stream, and a queue drain before every
raw-stream operation that follows a CubeCL launch (`RawExec::cube_launched`,
centralized in `raw_exec`). Scratch, RoPE tables and the small weight vectors
(norms, biases) live in pool tensors (`CubePool`); GEMM weights stay in the
`cuMemAlloc` arenas.

### Ported kernels (`tenferro_ext::cube_kernels`)

Laya: LayerNorm, bias/activation, GeGLU (MLX-exact erf) and the fused flash
attention (RoPE, online softmax, local-window/padding tile skipping). Jeff:
centered RMSNorm (masked), SiLU gate, the DeltaNet RMSNorm·silu(z) gate and
the chunked delta rule's `prepare` kernel. Embedding, marker pooling,
attention prep/attention of Jeff, conv, Q/K norm, gates, scans and the
triangular inverse stay NVRTC.

Portability work needed (found by running the same source on CubeCL's wgpu
runtime over lavapipe, `target/tenferro-mwe/21-cube-wgpu`, plane size 8,
32 KiB shared memory): no hard-coded 32-lane planes (norms use `PLANE_DIM` /
`UNIT_POS_PLANE`; launch geometry from `plane_size_max`), no reliance on
float tricks a shader compiler may fold (MLX's `+1.5·2^23` rounding replaced by
`round`), and attention shared memory exactly 32 KiB (the block-wide vote via
a shared flag pushed it 4 bytes over the limit and wgpu dropped the launch
silently; the votes are now block-uniform mask checks). After that all six
kernels match CPU references to ≤ 1.7e-6 on both wgpu (lavapipe) and CUDA.

### Results

Per kernel (nsys, GPU time per call; host GPU 0, see the note below):

| kernel | NVRTC | CubeCL |
|---|---:|---:|
| Laya LayerNorm, L8 / L512 | 12.3 / 21.1 µs | 13.3 / 18.4 µs |
| Laya attention, L8 / L512 | 10.7 / 311.5 µs | 11.3 / 291.4 µs |
| Laya GeGLU, L512 | 54.2 µs | 54.1 µs |
| Jeff RMSNorm, L8 / L256 | 12.2 / 12.9 µs | 12.9 / 13.4 µs |
| Jeff chunk prepare, L256 | 70.9 µs | 61.9 µs |
| Jeff RMS gate, L256 / SiLU gate, L256 | 31.9 / 34.6 µs | 32.6 / 34.5 µs |

(A non-portable first version with a 32-register warp-per-column LayerNorm
took 5.2 µs at L8, 2.4× faster than NVRTC, whose register array spills.)

End to end (warm median of 50 after 10 warmups, NVRTC and CubeCL alternated,
mean of two runs):

| row | raw NVRTC | raw CubeCL | CubeCL / NVRTC |
|---|---:|---:|---:|
| Jeff parcel trimmed L101 | 24.72 | 24.92 | 1.008 |
| Jeff parcel full L256 | 45.58 | 45.74 | 1.004 |
| Jeff synthetic L8 | 9.75 | 9.86 | 1.012 |
| Jeff synthetic L64 | 15.58 | 15.70 | 1.007 |
| Laya L8 B1 | 7.16 | 7.31 | 1.021 |
| Laya L64 B1 | 10.99 | 11.30 | 1.028 |
| Laya L8 B8 | 11.06 | 11.27 | 1.019 |
| Laya L93 B1 | 15.77 | 15.98 | 1.014 |
| Laya L93 B10 | 107.85 | 108.22 | 1.003 |
| Laya L512 B1 | 63.73 | 63.16 | 0.991 |
| Laya L512 B10 | 543.50 | 549.53 | 1.011 |

The 1–3% cost at small sizes is the queue drains (109 per Jeff forward, 127
per Laya forward, ~17 µs each, mostly overlapped with GPU work). Accuracy is
unchanged (Laya logits vs CPU ≤ 7.6e-6, vs NVRTC ≤ 5.2e-6; Jeff vs host oracle
≤ 8.6e-6 for both scans; production gates pass). First call: +1–2 s of CubeCL
JIT compilation.

Measurement note: the devcontainer lost GPU access during this work (NVML
"Unknown Error"), so the final per-kernel and end-to-end numbers were taken on
the host with CUDA 13 cuBLAS / NVRTC 12.8 libraries and the container's CUDA
13.0 headers; raw-path numbers match the container's (Laya L8B1 7.16 vs 7.2
ms).

### Blockers and caveats

- CubeCL's CUDA runtime needs a CUDA toolkit's headers at run time
  (`CUDA_PATH` or `/usr/local/cuda`, for `#include <cuda_runtime.h>` in the
  generated source); the NVRTC raw path needs only the driver, NVRTC and
  cuBLAS. Without `CUDA_PATH` the compile panics on CubeCL's device thread and
  the launches are dropped while the forward returns garbage (no error).
- No ordering contract between `#[cube]` launches on tenferro's client and
  raw-stream work in one scope (the drain is our convention); per-op scopes
  block (#2078-like).
- CubeCL can bind only pool tensors (not external device memory), whose
  memory is never returned to the driver without `memory_cleanup` (#2081).
- No `float4` shared loads or `__syncthreads_or` in the port (scalar loads,
  mask-based skips instead); dynamic shared memory above 48 KiB appears to be
  handled by CubeCL on CUDA (reported limit 101 KiB), unlike the raw seam
  (#2085).
- wgpu was checked on lavapipe only (no WebGPU/Metal hardware here); subgroup
  operations are an optional WebGPU feature.

### Recommendation

Hybrid, not yet the default: the CubeCL kernels are viable (correct, portable
after the fixes above, within 1–3% of NVRTC, faster for attention and chunk
prepare) and worth keeping as the portable path for the non-GEMM work, but the
NVRTC kernels stay the CUDA default until (1) tenferro defines ordering
between CubeCL launches and raw-stream work (or exposes a non-blocking scope),
(2) CubeCL's CUDA backend stops requiring toolkit headers or fails loudly, and
(3) a GEMM path exists for non-CUDA runtimes (cuBLAS is CUDA-only, so
portability today covers only the non-GEMM half).
