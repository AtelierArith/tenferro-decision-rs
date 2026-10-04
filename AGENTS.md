# AGENTS.md

Guidance for coding agents working in `tenferro-decision-rs`.

## Principle: prefer tenferro for model computation

Implement model computation with **tenferro operations** and keep the execution
path backend-agnostic. A tenferro-native path runs on every backend tenferro
supports, so CPU is today's target and CUDA / WebGPU / Apple GPU follow from
tenferro instead of bespoke device kernels. Custom host kernels and hand-rolled
GEMM/SIMD are a fallback, not the default.

Concretely:

- The tenferro forward (`forward_tenferro`) stays tenferro-native: embeddings,
  norms, attention, Gated DeltaNet, MLP, and the readout go through
  `EagerSession` / `tenferro-infer` ops, so new backends follow from tenferro.
  Engines select the forward per model (`LayaEngine` uses tenferro;
  `JeffEngine` defaults to the host path and offers a tenferro backend) because
  the tenferro path still round-trips through the host and is slower on CPU
  today — see `docs/agents/specs/docs/23_TENFERRO_NATIVE.md`.
- Prefer adding or extending tenferro ops over writing host loops. Use the
  self-hosted extension-op pattern (`tenferro-ext`, `tenferro-gated-delta`) and
  never modify tenferro-rs itself; record gaps in
  `docs/agents/specs/docs/19_TENFERRO_FEEDBACK.md`.
- Cache prepared tensors, plans, and workspaces instead of rebuilding them on
  every call. An `EagerTensor` keeps an `Arc<EagerRuntime>`, so prepared weight
  tensors can be reused across forwards that share the runtime — the tenferro
  path's dominant overhead is otherwise re-transposing and re-creating every
  weight tensor per call.
- The all-host `forward_reference` exists only as the correctness oracle
  (`docs/agents/specs/docs/05_TESTING_BENCHMARKS.md` §2). Keep it tenferro-free
  and cross-check the tenferro path against it; do not grow it as the production
  path.
- The `GatedDelta` extension op should execute through tenferro session ops
  (not the host kernel) once that is feasible, so it is backend-portable too.

## Constraints

- `decision-core` and `jev-client` must not depend on tenferro.
- `tenferro-gated-delta` is Jeff-only. Depend on tenferro's public crates only;
  pin revisions deliberately in `Cargo.toml`.
- Network/model access goes through `hf-fetch`; the inference crates stay
  offline and deterministic.
- Never modify the pinned tenferro-rs dependency. `extern/*` are submodules:
  change them only deliberately, and update the parent pin when you do.

## Checks before committing

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

The checkpoint-dependent tests (Laya/Jeff `real_checkpoint.rs`) skip when the
snapshot is absent; Jeff's is `#[ignore]`d. Do not commit absolute local paths
or model artifacts.
