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

## Performance

| # | Issue | Severity |
|---|---|---|
| [#1990](https://github.com/tensor4all/tenferro-rs/issues/1990) | CPU fused elementwise region is ~3.7× slower than eager per-op kernels for a long elementwise chain | medium |
| [#1992](https://github.com/tensor4all/tenferro-rs/issues/1992) | CPU eager `dot_general`: GEMM analysis recomputed per call (no plan cache) and the BLAS provider cannot execute linalg | medium |
| [#1995](https://github.com/tensor4all/tenferro-rs/issues/1995) | CPU eager decode forward: small-`m` GEMMs (faer) and internal layout copies (`structural::typed_copy_into_uninit`) dominate; GEMM providers are shape-dependent | medium |
| [#2003](https://github.com/tensor4all/tenferro-rs/issues/2003) | CPU eager `dot_general` is ~2.3× a plain host GEMM at decode (per-call overhead 1.7× over raw faer + faer-vs-`sgemm` kernel choice); no eager provider selection | medium |

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

## Positive findings (not filed)

- `triangular_solve(..., unit_diagonal = true)` fits the chunked Gated DeltaNet
  effective system exactly (the operator is applied and the diagonal ignored).
- `dot_general` batch dimensions are sufficient for attention / GQA.
- The eager session surface covers the inference primitives (norm, activations,
  softmax, RoPE, attention).
- Unsupported dtype/shape returns a typed error with no silent CPU fallback.
