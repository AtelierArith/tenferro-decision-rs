# Tenferro-Native Execution

**Status:** Laya and Jeff production forwards run tenferro-native; caching added  
**Date:** 2026-10-04  
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
but device parity and CUDA request integration remain pending (see
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
