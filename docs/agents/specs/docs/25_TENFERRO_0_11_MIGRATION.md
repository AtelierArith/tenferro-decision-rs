# tenferro-rs `0.11` Migration

**Status:** implemented  
**Date:** 2026-11-18  
**From:** git `471c4278` (workspace version `0.7.1`)  
**To:** crates.io `0.11.0` (tag `cef3746`)

The workspace now depends on the published `0.11` line instead of a git
revision: `Cargo.lock` fixes the exact release, and CI needs no git access for
tenferro. `11_TENFERRO_API_SURVEY.md` carries the re-surveyed API description;
this document records what moved and how the workspace adapted.

## Dependency and feature changes

| Item | Before | After |
|---|---|---|
| tenferro crates | git revision `471c4278` | `tenferro-* = "0.11.0"` from crates.io |
| CPU provider feature | `cpu-faer` | `native` (default; `blas` + a provider is the alternative) |
| WebGPU | experimental `webgpu` feature | removed upstream |
| Rust floor | 1.85 | 1.96 (tenferro's `rust-version`) |

`tenferro-einsum` and `tenferro-fft` split out of the runtime and are now
separate crates; this workspace does not use einsum equations or FFTs.

## The engine is caller-held

`CpuBackend` and `CudaBackend::new(device)` are gone. A `tenferro_cpu::CpuEngine`
(or `tenferro_gpu::cuda::CudaEngine::open(device)`) owns the thread pool, buffer
pool, and plan cache, and every operation borrows it as `&mut dyn Engine`.

- `EagerRuntime` stores no engine. `EagerRuntime::with_cpu_backend(engine)` and
  `EagerRuntime::with_eager_session(|s| …)??` became `EagerRuntime::new()` and
  `runtime.with_eager_session(&mut engine, |s| …)?` — one `Result`, one `?`.
- `with_backend_session(|s| …)` for direct tensor work became a plain
  `&mut engine` argument.
- `with_execution_scope` is gone; holding the engine covers it.
- `backend_session()` still lends the engine inside an eager session, so
  extension and CUDA code reach it through `tenferro_cpu::with_cpu_engine` /
  `tenferro_gpu::cuda::raw::with_cuda_engine`.

In the workspace: `LayaEngine` holds `engine: CpuEngine` (its methods are
`&mut self`); `JeffEngine` and `PreparedCpuModel` derive `Clone` and therefore
hold `engine: Arc<Mutex<CpuEngine>>`, shared by clones and serialized per call.

## Extension ops

`define_extension_runtime!` keeps its shape, but the execution callback now
takes `&mut dyn tenferro_tensor::Engine` instead of `&mut dyn BackendSession`,
`backend_bound = Engine`, and `extension_module::<CpuEngine>(engine_id)` names
the engine (`CpuBackend` is gone). `apply_eager_with_targeted_extension_in_session`
is unchanged.

Because `0.11` adds a standard `EagerSession::erf`, the inherent method shadows
the trait method of the same name. `tenferro-ext` renamed its MLX-compatible
`f32` kernel to `EagerSessionErfExt::erf_ext` and kept `gelu_erf` on that kernel,
so the Laya forward keeps the numerics it was validated against. `erf_ext` is
the only renamed public item.

## Shapes and reductions

`reduce_sum_squares` now takes `Option<&[usize]>` like every other reduction, so
composed norms pass `Some(&[axis])`.

## Traced programs

`TracedTensor` no longer has context-free constructors: a `TraceContext` issues
inputs (`trace.input_concrete_shape`) and constants
(`trace.from_vec_col_major`). `tenferro-gated-delta::traced_layer` threads the
context through `prepare_traced_weights`, `delta_layer_traced`, `input_x`, and
`input_mask`.

Compiled programs changed entry points: `prepare_compiled` / `run_prepared` /
`run_compiled` became `Runtime::prepare(&program, &reads, &[target])`,
`Runtime::run(&mut [&mut engine], &prepared, &reads)` (returning `Output`s that
`into_tensor(engine)` converts), and `Runtime::warm_up` for the first execution.

## CUDA raw seam

`tenferro_gpu::cuda::raw` keeps the parts only tenferro can provide — the stream,
allocations, transfers, and resource caches — and drops the launch helpers:
`Module`, `Function`, `NvrtcOptions`, `KernelArg`, `LaunchConfig`,
`Session::{compile_nvrtc, launch, runtime_identity}`, and
`with_cuda_exec_session` are gone. A downstream crate compiles with
`cudarc::nvrtc` and launches with `cudarc::driver::result`, using
`tenferro_gpu::cuda::cudarc`.

`tenferro-gated-delta` restores the small surface its DeltaNet stages need in
`src/cuda/driver.rs`: `CudaSession` wraps `raw::Session` with `compile_nvrtc`,
`launch`, and `runtime_identity`, and `with_cuda_exec_session` keeps the
`cuda.runtime()` / `cuda.with_raw(op, …)` shape the layer adapters already use.
The kernel source, launch geometry, and ABI are unchanged.

This lane needs CUDA hardware and NVRTC to run; CI builds it (`--features cuda`)
but the parity tests stay `#[ignore]`d.

## Verification

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo check -p tenferro-ext -p tenferro-gated-delta --features cuda --all-targets
cargo check -p jeff-infer -p laya-infer --features hub --all-targets
cargo check -p jev-client --features http --all-targets
```

The checkpoint-dependent tests still skip without a snapshot, as before.
