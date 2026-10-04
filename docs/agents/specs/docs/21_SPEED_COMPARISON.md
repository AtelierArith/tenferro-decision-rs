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

- Julia: OpenBLAS 0.3.30 with **8 BLAS threads** (the default, = physical
  cores); `Threads.nthreads() = 1` (started without `-t`). JeffClient's
  `initialize_cpu!` explicitly sets the same 8 threads when
  `Threads.nthreads() == 1`.
- Rust: `rayon` parallelizes the GEMM kernels; the default pool uses all 16
  logical CPUs. The tables below use the matched **`RAYON_NUM_THREADS=8`**;
  using 16 threads changes the results by only a few percent (the work is
  memory-bandwidth-bound past the 8 physical cores).

The best Julia configuration differs by model, so each table uses its fastest
(and also reports the alternative):

- **Laya** is fastest with the default **BLAS=8, Julia threads=1**: its `Linear`
  layers are a single `transpose(W) * X` GEMM, so the work lives in BLAS.
  Running `-t 8` with `OPENBLAS_NUM_THREADS=1` makes it *slower* (L8 287 ms vs
  94 ms) because `Threads.@threads` only covers LayerNorm / GeLU-gate /
  attention, not the projections.
- **Jeff** is fastest with **`-t 8`, BLAS=1**: its `parallel_projections`,
  `parallel_full_heads`, and `recurrent_delta` fused kernels parallelize the
  work with Julia tasks. `initialize_cpu!` sets BLAS=8 when `nthreads == 1`
  (the default), which is 4–7× *slower* for this hybrid model.

## Best vs best

Fastest configuration per side (Julia's best per model as above; Rust = the
better of the host and cached-tenferro paths), 8 threads, production
checkpoints.

| model | shape | Julia best | Rust best | Rust / Julia |
|---|---|---:|---:|---:|
| Laya | L8 B1 | 94.0 ms (BLAS=8) | 160 ms (tenferro cached) | 1.70× |
| Laya | L64 B1 | 339.1 ms | 638 ms (host+GEMM+rayon) | 1.88× |
| Laya | L8 B8 | 309.3 ms | 511 ms (host+GEMM+rayon) | 1.65× |
| Laya | model load | 1535 ms | 6591 ms | 4.29× |
| Jeff | L8 | 128.5 ms (`-t 8`) | 159.5 ms (host+GEMM+rayon) | 1.24× |
| Jeff | L16 | 119.8 ms | 210.4 ms | 1.76× |
| Jeff | L64 | 198.9 ms | 608.3 ms | 3.06× |
| Jeff | model load | 5442 ms | 4699 ms | **0.86×** |

- Laya's Rust best is the cached tenferro path at L8B1; at L64/L8B8 the tenferro
  cache was not measured, so the host path is listed.
- Jeff's Rust best is always the host path; the cached tenferro path is ~10×
  slower on CPU (see the Jeff section).
- Rust loads Jeff faster than Julia and Laya slower.

## Laya

Checkpoint: `convaiinnovations/laya@main` (commit
`7b928d828b7b0e022f929d9bd2e44165aa270148`, ~843 MB), hidden 1024, 28 layers.

| shape | Julia CPU | Rust host (naive) | Rust host (+GEMM) | Rust host (+GEMM+rayon 8) | Rust tenferro |
|---|---|---|---|---|---|
| L8 B1 | 94.0 ms | 2692 ms (~29×) | 214.1 ms (~2.3×) | 215.7 ms (~2.3×) | 563 ms (~6×) |
| L64 B1 | 339.1 ms | 21443 ms (~63×) | 852.8 ms (~2.5×) | 638.3 ms (~1.9×) | — |
| L8 B8 | 309.3 ms | 21363 ms (~69×) | 704.7 ms (~2.3×) | 511.2 ms (~1.7×) | — |
| model load | 1535 ms | 6566 ms | 6620 ms | 6637 ms | — |

- The `+GEMM` column routes every host projection through `cpu-kernels`
  (`matrixmultiply::sgemm`): a **12–30×** speedup over the naive loops.
- `+GEMM+rayon 8` also parallelizes each GEMM across output rows (`rayon`,
  pinned to 8 threads); the smallest shape is already dominated by streaming the
  weights, so it barely moves.
- Rust is now ~1.7–2.3× behind Julia's multithreaded BLAS.
- Julia `-t 8` with `OPENBLAS_NUM_THREADS=1` is slower (287 / 635 / 637 ms), so
  the default BLAS=8 configuration is Julia's best for Laya.
- `forward_tenferro` rebuilds every weight tensor per call (563 ms in the table
  above); with a reused `TensorCache` it drops to **160 ms**, faster than the
  host+GEMM path. `LayaEngine` runs this cached tenferro path by default
  (`23_TENFERRO_NATIVE.md`).

Closing this gap is tracked in
[issue #2](https://github.com/AtelierArith/tenferro-decision-rs/issues/2).

## Jeff

Checkpoint: `mstrasser/Jeff-Qwen3.5-0.8B@0f212b3e72acb4dde3f7da61e925d6ab7f819990`
(~1.7 GB), hidden 1024, 24 layers (18 Gated DeltaNet + 6 full), vocab 248320.
Warmup 1, 3 iterations, median.

| shape | Julia (BLAS=8, default) | Julia (-t 8, BLAS=1) | Rust host (naive) | Rust host (+GEMM) | Rust host (+GEMM+rayon 8) | Rust tenferro |
|---|---|---|---|---|---|---|
| L8 | 842 ms | **128.5 ms** | 8657 ms (~10×) | 463 ms (3.6×) | 261 ms (2.0×) | 6662 ms |
| L16 | 893 ms | **119.8 ms** | 18405 ms (~21×) | 678 ms (5.7×) | 421 ms (3.5×) | — |
| L64 | 716 ms | **198.9 ms** | 75564 ms (~105×) | 2037 ms (~10×) | 1437 ms (7.2×) | — |
| model load | 11756 ms | 5442 ms | 4297 ms | 4507 ms | 4542 ms | — |

- Julia's fastest configuration is `-t 8` (task-parallel, BLAS=1); the default
  BLAS=8 is 4–7× slower for this hybrid model. The ratios above are relative to
  that best configuration.
- Rust's `+GEMM+rayon` is 18–37× faster than the naive loops; against Julia's
  best it is 2.0× (L8) / 3.5× (L16) / 7.2× (L64) slower.
- The gap grows with length: the Gated DeltaNet recurrent scan and the remaining
  single-threaded elementwise/normalization work dominate at L64.
- Rust model loading is ~1.2–2.6× faster than Julia (4297–4542 ms vs
  5442–11756 ms).
- `forward_tenferro` is unchanged by the host kernels and still ~25× slower
  than the host path because it rebuilds every weight tensor per call.
