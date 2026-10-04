# Speed Comparison: Julia vs Rust

**Status:** in progress (Laya measured; Jeff pending)  
**Date:** 2026-10-04

Fair-comparison rules (`05_TESTING_BENCHMARKS.md`, and the JeffClient.jl
`AGENTS.md`): same checkpoint, same precision, same prepared inputs, model
loading excluded from the forward, warmup before measurement, wall clock.

Methods:

- Julia: `tools/bench_laya_real.jl` / `tools/bench_jeff_real.jl` (host arrays).
- Rust: `cargo run --release -p laya-infer --example bench_laya -- <dir>` and
  `crates/jeff-infer/examples/bench_jeff.rs`.
- Both use fixed prepared batches; warmup 2, 5 iterations, median reported.

Machine: Intel Core i9-9900K (16 logical CPUs), macOS 15.8.1, x86_64,
rustc 1.99.0, Julia 1.13.1 (OpenBLAS; no Accelerate forwarding).

## Laya

Checkpoint: `convaiinnovations/laya@main` (commit
`7b928d828b7b0e022f929d9bd2e44165aa270148`, ~843 MB), hidden 1024, 28 layers.

| shape | Julia CPU | Rust host | Rust tenferro |
|---|---|---|---|
| L8 B1 | 94.0 ms | 2692 ms (~29×) | 845 ms (~9×) |
| L64 B1 | 339.1 ms | 21443 ms (~63×) | — |
| L8 B8 | 309.3 ms | 21363 ms (~69×) | — |
| model load | 1535 ms | 6566 ms | — |

- `forward_reference` is naive host loops with no BLAS/SIMD; Julia uses OpenBLAS
  for the projections.
- `forward_tenferro` is faster than the host path (optimized `dot_general`) but
  still rebuilds every weight/constant tensor on each call, so it remains ~9×
  slower.

Closing this gap is tracked in
[issue #2](https://github.com/AtelierArith/tenferro-decision-rs/issues/2).

## Jeff

Checkpoint: `mstrasser/Jeff-Qwen3.5-0.8B@0f212b3e72acb4dde3f7da61e925d6ab7f819990`
(~1.7 GB), hidden 1024, 24 layers (18 Gated DeltaNet + 6 full), vocab 248320.
Pending: run `tools/bench_jeff_real.jl` (`NativeBackend` + `logits`) and
`crates/jeff-infer/examples/bench_jeff.rs`, then fill in the table.
