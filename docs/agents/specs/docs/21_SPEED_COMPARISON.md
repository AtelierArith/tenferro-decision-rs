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

| model | shape | Julia best | Rust host_opt | Rust tenferro (HR) | host_opt / Julia | tenferro / Julia |
|---|---|---:|---:|---:|---:|---:|
| Laya | L8 B1 | 88.7 ms (BLAS=8) | 93.3 ms | 158.8 ms (cached) | 1.05× | 1.79× |
| Laya | L64 B1 | 213.0 ms | 434.2 ms | — | 2.04× | — |
| Laya | L8 B8 | 211.9 ms | 284.0 ms | — | 1.34× | — |
| Laya | model load | 1516 ms | 6463 ms | — | 4.26× | — |
| Jeff | L8 | 137.5 ms (`-t 8`) | 106.8 ms | 257.4 ms | **0.78×** | 1.87× |
| Jeff | L16 | 143.3 ms | 124.3 ms | 293.8 ms | **0.87×** | 2.05× |
| Jeff | L64 | 238.3 ms | 248.7 ms | 408.7 ms | 1.04× | 1.72× |
| Jeff | model load | 5495 ms | 4369 ms | — | **0.79×** | — |

- Laya's Rust best is the host path at L8B1 (row-blocked `matrixmultiply`, now
  blocking over `out_dim` so decode parallelizes) and the cached tenferro path is
  close; L64/L8B8 still favor Julia's multithreaded BLAS.
- Jeff's Rust best is `host_opt`. It is **faster than Julia at L8/L16** (0.78× /
  0.87×) and within ~4% at L64. The wins are the `pulp` runtime-dispatched SIMD
  DeltaNet scan and the row-blocked `matrixmultiply` projections for Jeff's
  `(out_dim, length)` layout (which beat Accelerate). The tenferro
  (host-recurrent DeltaNet) path shares the improved kernels and is ~1.7–2.1×.
- Rust loads Jeff faster than Julia and Laya slower. Host numbers vary ~±20%
  run-to-run with machine load/thermals.

## Laya

Checkpoint: `convaiinnovations/laya@main` (commit
`7b928d828b7b0e022f929d9bd2e44165aa270148`, ~843 MB), hidden 1024, 28 layers.

| shape | Julia CPU (BLAS=8) | Rust host | Rust tenferro (cached) |
|---|---|---|---|
| L8 B1 | 88.7 ms | 93.3 ms (1.05×) | 158.8 ms (1.79×) |
| L64 B1 | 213.0 ms | 434.2 ms (2.04×) | — |
| L8 B8 | 211.9 ms | 284.0 ms (1.34×) | — |
| model load | 1516 ms | 6463 ms | — |

- The host path routes every projection through `cpu-kernels`, now a portable
  rayon **row-blocked `matrixmultiply`** for both kernels (the Accelerate
  `cblas_sgemm` path was dropped; see `22_CPU_KERNEL_OPTIMIZATION.md`). Laya's
  `x·Wᵀ` kernel now blocks over `out_dim`, so its decode case improved further:
  L8 B1 **106 → 93 ms** (1.19× → 1.05× of Julia).
- Rust is ~1.05–2.0× behind Julia's multithreaded BLAS; L64 B1 (prefill-ish) is
  the remaining gap.
- `forward_tenferro` rebuilds every weight tensor per call; with a reused
  `TensorCache` it is **158.8 ms** at L8B1. `LayaEngine` runs this cached
  tenferro path by default (`23_TENFERRO_NATIVE.md`).

Closing this gap is tracked in
[issue #2](https://github.com/AtelierArith/tenferro-decision-rs/issues/2).

## Jeff

Checkpoint: `mstrasser/Jeff-Qwen3.5-0.8B@0f212b3e72acb4dde3f7da61e925d6ab7f819990`
(~1.7 GB), hidden 1024, 24 layers (18 Gated DeltaNet + 6 full), vocab 248320.
Warmup 3, 15 iterations, min.

| shape | Julia (-t 8) | Rust oracle `forward_reference` | Rust `host_opt` | host_opt / Julia |
|---|---|---|---|---|
| L8 | 137.5 ms | 126.5 ms (0.92×) | 106.8 ms | **0.78×** |
| L16 | 143.3 ms | 137.1 ms (0.96×) | 124.3 ms | **0.87×** |
| L64 | 238.3 ms | 379.6 ms (1.59×) | 248.7 ms | **1.04×** |
| model load | 5495 ms | 4369 ms | — | — |

- Julia's fastest configuration is `-t 8` (task-parallel, BLAS=1); the default
  BLAS=8 is 4–7× slower for this hybrid model.
- `forward_reference` is the correctness oracle; `host_opt`
  (`JeffBackend::HostOpt`, the default) is the optimized host path. With the
  `pulp` runtime-dispatched SIMD DeltaNet scan and the **row-blocked
  `matrixmultiply`** projections (which beat Accelerate for Jeff's
  `(out_dim, length)` layout), `host_opt` is now **faster than Julia at L8/L16**
  and within ~4% at L64.
- Against Julia's best, the optimized host path is 0.78× (L8) / 0.87× (L16) /
  1.04× (L64).
- The tenferro path runs the fused host recurrent DeltaNet via the `GatedDelta`
  extension op (`DeltaKernel::HostRecurrent`, the default). It shares the
  improved host kernels; `TensorNative` (tensor-only, for portability) remains
  slower. See `22_CPU_KERNEL_OPTIMIZATION.md`.
- Rust model loading is ~1.3× faster than Julia for Jeff.
- `forward_tenferro` is rebuilt from tenferro ops with a reused `TensorCache`.
  `JeffEngine` defaults to `JeffBackend::Auto`, which picks `host_opt` for
  sequences of at least 16 tokens and the oracle below that; `HostOpt` and
  `Tenferro` can be selected explicitly (`JeffBackend::{Auto, Host, HostOpt,
  Tenferro}`).

## Apple silicon (M2 Max) + tenferro extension-op pass

**Date:** 2026-10-05. Re-measured with `tools/bench_compare.sh` and the
per-shape gap benches on an Apple M2 Max (12 cores, macOS 26.5, Julia 1.13.1,
rustc 1.98.1, release), 8 threads both sides, warmup 5 / 30 iterations, median.

**Julia BLAS correction.** On Apple Silicon, `QwenDecisionCore` used to
`import AppleAccelerate` unconditionally, so `using JeffClient` forwarded BLAS
to Accelerate even in the default env — the "OpenBLAS" Jeff row was really
Accelerate with `BLAS=8`. Current `QwenDecisionCore` `main` makes that opt-in,
so the default Jeff env is OpenBLAS + `BLAS=1`; run `Pkg.update` and the Jeff
Julia L8 row drops **114.7 → 68.9 ms**. The two Julia configs only differ for
Laya by default.

### Jeff

| length | Julia (OpenBLAS) | Julia (Accelerate) | Rust oracle | Rust host_opt | Rust tenferro |
|---:|---:|---:|---:|---:|---:|
| L8 | 68.9 | 114.7 | 67.0 | 70.6 | 78.2 |
| L16 | 90.9 | 112.1 | 90.8 | 90.2 | 98.1 |
| L64 | 216.6 | 134.9 | 266.5 | 208.6 | 212.0 |

tenferro / host_opt: 1.12× (L8), 1.06× (L16), 1.02× (L64). Julia's best is
OpenBLAS for L8/L16 and Accelerate for L64.

### Laya

| shape | Julia (OpenBLAS) | Julia (Accelerate) | Rust host | Rust tenferro |
|---|---:|---:|---:|---:|
| L8 B1 | 90.7 | 88.4 | 45.2 | 49.8 |
| L16 B1 | 120 | 91.5 | 67.6 | 69.6 |
| L64 B1 | 286.3 | 88.5 | 249.3 | 169.0 |
| L8 B8 | 250.8 | 85.9 | 173.2 | 158.3 |

tenferro / host: 1.08× (L8 B1), 1.03× (L16 B1), 0.68× (L64 B1), 0.93× (L8 B8).

### tenferro-native tuning

The tenferro forward now reaches the host kernels through self-hosted
`cpu-kernels` extension ops (`tenferro-ext`): **Laya** routes `linear`/bias,
feature-first LayerNorm, GeGLU, and the whole `split + RoPE + masked attention`
block through one op each; **Jeff** routes `linear`, the full-attention block
(RMSNorm ×2 + partial RoPE ×2 + causal masked attention + sigmoid gate), the
feature-last RMSNorm, and gated SiLU. Effect on tenferro / host: Laya L8 B1
1.53 → 1.08×, L64 1.07 → 0.68×, L8 B8 1.52 → 0.93×; Jeff 1.20 → 1.12× (L8),
1.17 → 1.06× (L16), 1.11 → 1.02× (L64). A standalone masked-attention op and a
standalone `split_qkv` op were tried and reverted (no net win at the small
decode shapes the gap is in).

`matrixmultiply` note: the Jeff `linear` kernel must feed a column-major `A`
(`rsa = 1`) and a row-major `B` (`csb = 1`) — the column-major activations plus
the raw row-major weight give exactly that. A single-threaded or row-major-`A`
variant is 3–5× slower, which is why the host path parallelizes over output
columns.
