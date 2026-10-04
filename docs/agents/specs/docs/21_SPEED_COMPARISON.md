# Speed Comparison: Julia vs Rust

**Status:** measured (Laya and Jeff)  
**Date:** 2026-10-04

Fair-comparison rules (`05_TESTING_BENCHMARKS.md`, and the JeffClient.jl
`AGENTS.md`): same checkpoint, same precision, same prepared inputs, model
loading excluded from the forward, warmup before measurement, wall clock.

Methods:

- Julia: `tools/bench_laya_real.jl` / `tools/bench_jeff_real.jl` (host arrays).
- Rust: `cargo run --release -p laya-infer --example bench_laya -- <dir>` and
  `crates/jeff-infer/examples/bench_jeff.rs`.
- Both use fixed prepared batches; warmup/iteration counts are noted per
  section, and the median is reported.

Machine: Intel Core i9-9900K (8 physical / 16 logical CPUs), macOS 15.8.1,
x86_64, rustc 1.99.0, Julia 1.13.1.

Thread configuration:

- Julia is run with `-t 8`. **Laya** uses the default **8 BLAS threads** (its
  work is a few big `transpose(W) * X` GEMMs, so it lives in OpenBLAS).
  **Jeff** is run with `OPENBLAS_NUM_THREADS=1` (task-parallel fused kernels;
  BLAS=8 is 4–7× *slower* for this hybrid model). Julia `initialize_cpu!` sets
  BLAS=8 when `Threads.nthreads() == 1`, which is the default.
- Rust: `rayon` parallelizes the GEMM kernels; the default pool uses all 16
  logical CPUs. The tables below use the matched **`RAYON_NUM_THREADS=8`**;
  using 16 threads changes the results by only a few percent (the work is
  memory-bandwidth-bound past the 8 physical cores).

The best Julia configuration differs by model, so each table uses its fastest
(and also reports the alternative):

- **Laya** is fastest with the default **BLAS=8** (Julia threads=1): its `Linear`
  layers are a single `transpose(W) * X` GEMM, so the work lives in BLAS.
- **Jeff** is fastest with **`-t 8`, BLAS=1**: its `parallel_projections`,
  `parallel_full_heads`, and `recurrent_delta` fused kernels parallelize the
  work with Julia tasks. BLAS=8 (the default) is 4–7× *slower* for this hybrid
  model.

## Best vs best

Fastest configuration per side (Julia's best per model as above; Rust = the
better of the host and cached-tenferro paths), 8 threads, production
checkpoints. Re-measured 2026-10-04 with the host GEMM on
**Accelerate** (`cpu-kernels` uses `cblas_sgemm` on macOS above a threshold),
which is why the host path is much faster than earlier tables.

| model | shape | Julia best | Rust host | Rust tenferro (HR) | host / Julia | tenferro / Julia |
|---|---|---:|---:|---:|---:|---:|
| Laya | L8 B1 | 88.7 ms (BLAS=8) | 105.8 ms | 175.6 ms (cached) | 1.19× | 1.98× |
| Laya | L64 B1 | 213.0 ms | 428.3 ms | — | 2.01× | — |
| Laya | L8 B8 | 211.9 ms | 279.6 ms | — | 1.32× | — |
| Laya | model load | 1516 ms | 6463 ms | — | 4.26× | — |
| Jeff | L8 | 137.5 ms (`-t 8`) | 163.5 ms | 287.4 ms | 1.19× | 2.09× |
| Jeff | L16 | 143.3 ms | 201.7 ms | 351.0 ms | 1.41× | 2.45× |
| Jeff | L64 | 238.3 ms | 568.8 ms | 598.9 ms | 2.39× | 2.51× |
| Jeff | model load | 5495 ms | 4369 ms | — | **0.79×** | — |

- Rust host is now within ~1.2× (L8) of Julia's best and ~2.0–2.4× behind at
  longer lengths. The tenferro (host-recurrent DeltaNet) path is ~2.1× at L8 and
  roughly host-parity at L64 for Jeff; Laya's cached tenferro path is ~2.0×.
- The gap grows with length: the DeltaNet recurrent scan and the elementwise /
  normalization work dominate at L64.
- Rust loads Jeff faster than Julia and Laya slower. Host numbers vary ~±20%
  run-to-run with machine load/thermals.

## Laya

Checkpoint: `convaiinnovations/laya@main` (commit
`7b928d828b7b0e022f929d9bd2e44165aa270148`, ~843 MB), hidden 1024, 28 layers.

| shape | Julia CPU (BLAS=8) | Rust host (Accelerate) | Rust tenferro (cached) |
|---|---|---|---|
| L8 B1 | 88.7 ms | 105.8 ms (1.19×) | 175.6 ms (1.98×) |
| L64 B1 | 213.0 ms | 428.3 ms (2.01×) | — |
| L8 B8 | 211.9 ms | 279.6 ms (1.32×) | — |
| model load | 1516 ms | 6463 ms | — |

- The host path routes every projection through `cpu-kernels`, which on macOS
  uses **Accelerate `cblas_sgemm`** for large GEMMs (and `matrixmultiply`
  otherwise). This is the current host, replacing the earlier
  `matrixmultiply`-only numbers — L8 B1 dropped from 216 ms to 106 ms and
  L64 B1 from 638 ms to 428 ms.
- Rust is now ~1.2–2.0× behind Julia's multithreaded BLAS.
- `forward_tenferro` rebuilds every weight tensor per call; with a reused
  `TensorCache` it is **175.6 ms** at L8B1. `LayaEngine` runs this cached
  tenferro path by default (`23_TENFERRO_NATIVE.md`), though the Accelerate host
  path is now faster at L8B1.

Closing this gap is tracked in
[issue #2](https://github.com/AtelierArith/tenferro-decision-rs/issues/2).

## Jeff

Checkpoint: `mstrasser/Jeff-Qwen3.5-0.8B@0f212b3e72acb4dde3f7da61e925d6ab7f819990`
(~1.7 GB), hidden 1024, 24 layers (18 Gated DeltaNet + 6 full), vocab 248320.
Warmup 1, 3 iterations, median.

| shape | Julia (-t 8) | Rust host (Accelerate) | Rust tenferro `HostRecurrent` | Rust tenferro `TensorNative` |
|---|---|---|---|---|
| L8 | 137.5 ms | 163.5 ms (1.19×) | 287.4 ms (2.09×) | 361 ms (2.62×) |
| L16 | 143.3 ms | 201.7 ms (1.41×) | 351.0 ms (2.45×) | 407 ms (2.84×) |
| L64 | 238.3 ms | 568.8 ms (2.39×) | 598.9 ms (2.51×) | 676 ms (2.84×) |
| model load | 5495 ms | 4369 ms | — | — |

- Julia's fastest configuration is `-t 8` (task-parallel, BLAS=1); the default
  BLAS=8 is 4–7× slower for this hybrid model. Julia `-t 8` numbers are the
  min of 15 iterations.
- The Rust host now uses Accelerate for its GEMMs; against Julia's best it is
  ~1.2× (L8) / 1.4× (L16) / 2.4× (L64).
- The tenferro path runs the fused host recurrent DeltaNet via the `GatedDelta`
  extension op (`DeltaKernel::HostRecurrent`, the default); it is ~2.1–2.5×
  Julia, and roughly host-parity at L64. `TensorNative` (tensor-only, for
  portability) is ~2.6–2.8×. See `22_CPU_KERNEL_OPTIMIZATION.md` for the
  remaining tenferro-internal gap ([#1995](https://github.com/tensor4all/tenferro-rs/issues/1995)).
- The gap grows with length: the Gated DeltaNet recurrent scan and the
  elementwise/normalization work dominate at L64.
- Rust model loading is ~1.3× faster than Julia for Jeff.
- `forward_tenferro` is rebuilt from tenferro ops with a reused `TensorCache`.
  `JeffEngine` defaults to the host backend and offers the tenferro one
  (`JeffBackend::Tenferro`) for portability.
