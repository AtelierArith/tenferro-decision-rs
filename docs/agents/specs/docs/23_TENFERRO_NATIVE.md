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
  gates, chunked scan, output projection). Jeff's tenferro path calls it
  directly, so there are no host round-trips between layers.
- The engines' `decide` / `logits` / `row_logits` now take `&mut self` for the
  cache (the eager session API requires a `Send` closure, so a `RefCell` borrow
  cannot cross it).

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

| path | L8 |
|---|---:|
| host + GEMM + rayon | 177 ms |
| tenferro, fresh cache each call | 4677 ms |
| tenferro, reused cache, old host-round-trip chunked path | 1693 ms |
| tenferro, reused cache, tensor-native (head-sequential) | 496 ms |
| **tenferro, reused cache, tensor-native, head-batched** | **342 ms** |

Caching plus the tensor-native rewrite cut the tenferro path ~5× (1693 → 342 ms),
to ~1.9× the host path on CPU. The remaining gap is the tenferro eager-op /
small-GEMM overhead; the payoff is a single backend-portable code path.

## Backend selection

Because the two paths differ by model, the engines let the caller choose:

- `LayaEngine` runs the tenferro forward by default (it is faster).
- `JeffEngine` defaults to `JeffBackend::Host` and offers
  `JeffBackend::Tenferro` (via `with_backend`) for backend portability / future
  GPU execution.

The host forwards remain the correctness oracle.

## `GatedDelta` extension op

Making the extension op execute through tenferro session ops is **not feasible**
with the current extension API: `execute_in_session` receives only a
`&mut dyn BackendSession`, and `ExtensionExecutionContext` exposes just the
backend and caches — no tensor-op helpers (`dot_general`, `triangular_solve`,
…). The op therefore keeps its CPU fused kernel. This is not a GPU blocker for
the production path, which already runs Gated DeltaNet through
`tensor_layer::delta_layer_tenferro_native` (session ops).

## Next

- Apply the same prepared-tensor approach to any remaining per-call host work
  (e.g. attention mask construction) and to future backends.
- When tenferro exposes device backends, weight tensors prepared the same way
  can be kept device-resident.
