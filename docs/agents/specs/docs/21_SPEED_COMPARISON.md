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

| length | Julia (OpenBLAS) | Julia (Accelerate) | Rust (opt) | Rust (tenferro-rs) |
|---:|---:|---:|---:|---:|
| L8 | 68.9 | 114.7 | 70.6 | 78.2 |
| L16 | 90.9 | 112.1 | 90.2 | 98.1 |
| L64 | 216.6 | 134.9 | 208.6 | 212.0 |

tenferro / host_opt: 1.12× (L8), 1.06× (L16), 1.02× (L64). Julia's best is
OpenBLAS for L8/L16 and Accelerate for L64.

### Laya

| shape | Julia (OpenBLAS) | Julia (Accelerate) | Rust (opt) | Rust (tenferro-rs) |
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

## Original Python / PyTorch comparison

`tools/bench_python_real.py` runs the upstream forward directly, including the
trained readout and (for Laya) action head. It uses the same fixed prepared
batches as the Rust and Julia scripts, float32 on CPU, inference mode, and an
explicit thread count. Checkpoint loading is timed separately. The L8 B1 logits
must pass a comparison with the committed production Julia reference before
benchmark results are accepted. This measures forward latency; tokenization,
prompt rendering, and answer conversion are excluded on all sides.

Pass `--python-bin`, `--python-laya-source`, and `--python-jeff-source` to
`tools/bench_compare.sh` to include the Python rows in its Markdown report.
The source arguments refer to local upstream Git checkouts, not installed
package aliases; the JSON captures their revisions and the PyTorch/Transformers
versions. The Python process forces offline Hub access.

The Rust benchmark examples also capture `bench-suite` machine/build metadata
and the actual Rayon pool size in their JSON output. Keep these fields with
new measurements so comparisons can be traced to their execution environment.

Example (in an isolated Python environment with CPU torch, torchvision,
transformers, safetensors, Pillow, and NumPy installed):

```sh
python tools/bench_python_real.py laya "$LAYA_CHECKPOINT" \
  --source "$LAYA_SOURCE" --threads 8 --warmup 2 --iters 5
python tools/bench_python_real.py jeff "$JEFF_CHECKPOINT" \
  --source extern/JeffClient.jl/extern/jeff --threads 8 --warmup 2 --iters 5
```

Laya upstream: https://github.com/NandhaKishorM/laya (the original PyTorch
implementation, rather than the MLX port). Jeff upstream:
https://github.com/firelex/jeff (the pinned nested submodule).

### Ryzen 9 PRO 8945HS CPU measurement (2026-10-08)

Linux x86_64, 8 physical / 16 logical CPUs; 8 PyTorch/Rayon threads, float32,
2 warmup forwards and 5 measured forwards, median. Runs were sequential;
model loading and input preparation are excluded. Raw samples and dependency
versions: `fixtures/bench-python-cpu-2026-10-08/`.

| model | shape | Python PyTorch (ms) | Rust host (ms) | Rust / Python |
|---|---|---:|---:|---:|
| Laya | L8 B1 | 86.9 | 88.0 | 1.01× |
| Laya | L64 B1 | 135.3 | 370.6 | 2.74× |
| Laya | L8 B8 | 158.3 | 274.4 | 1.73× |
| Jeff | L8 B1 | 132.6 | 109.6 | 0.83× |
| Jeff | L16 B1 | 143.1 | 112.3 | 0.78× |
| Jeff | L64 B1 | 196.4 | 175.8 | 0.90× |

The Rust Laya column above is the host correctness path. `LayaEngine` uses
cached tenferro: the same report records 105.0 ms for L8 B1 versus Python's
86.9 ms (1.21×), with only two measured native forwards. That native result
is preliminary; the host rows do not describe production LayaEngine latency.
Jeff's HostOpt rows match its default engine backend.

PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, Transformers 5.17.0,
safetensors 0.8.0, Python 3.12.3. Jeff uses the Transformers CPU reference
DeltaNet kernels (`causal_conv1d` and `flash-linear-attention` are absent);
this comparison does not measure the optional CUDA kernels or torch.compile.
Laya uses the upstream eager backend. L8 B1 logit errors against the committed
Julia references: Laya 2.44e-6, Jeff 6.68e-6. Laya's wider/longer shapes favor
PyTorch's CPU kernels; these results do not establish that the performance
issue is resolved. The existing Apple measurements above are from a different
machine and should not be combined into ratios with these Python rows.

### Cached tenferro Laya after parallel GeGLU (2026-10-08)

On the same Ryzen CPU, float32, eight Rayon/Julia workers, warmup 2 and median
of 5, token-column GeGLU parallelism improves production Laya's tenferro
forward. Julia was remeasured on this machine with eight BLAS threads; these
are not the earlier Apple-platform timings. Runs were sequential.

| shape | tenferro before (ms) | tenferro after (ms) | Julia (ms) | after / Julia |
|---|---:|---:|---:|---:|
| L8 B1 | 113.1 | 101.5 | 72.6 | 1.40× |
| L64 B1 | 297.3 | 220.2 | 122.0 | 1.80× |
| L8 B8 | 284.5 | 211.3 | 117.9 | 1.79× |

Raw reports: `fixtures/bench-geglu-cpu-2026-10-08/`. The after report includes
`bench-suite` machine/build metadata; `source_patch` identifies the uncommitted
kernel change applied to its recorded Git revision at measurement time. The
baseline report predates metadata capture in the per-shape example. A second
after run measured L64 B1 223.5 ms and L8 B8 243.7 ms, so batch timing has some
run-to-run variation. These results demonstrate a production-path improvement,
while #2's remaining Laya performance gap stays open.

The same-machine Julia Jeff remeasurement is included in this directory:
76.2/97.5/178.5 ms at L8/L16/L64, eight Julia workers and one BLAS thread
(selected by the upstream native backend). Rust HostOpt's Python-comparison
report above records 109.6/112.3/175.8 ms; the relative Julia gap is therefore
shape-dependent, rather than a uniform language/runtime advantage.

An optional system OpenBLAS provider was also measured on this machine with
`cpu-kernels/openblas`: at eight BLAS/Rayon threads, cached Laya L64 B1 was
216.9 ms (repeat 215.1), and L8 B8 was 201.6 ms (repeat 201.7), against a
contemporaneous portable baseline of 225.0/216.6 ms. This is a modest additional
4%/7% improvement, not closure of the Julia/Python gap. One BLAS thread was
much slower (618.4/599.3 ms), so thread settings matter. The optional provider
uses the existing tenferro extension path and requires an installed LP64
OpenBLAS; the default remains portable. Exact conditions and report links are
in `22_CPU_KERNEL_OPTIMIZATION.md`.


On 2026-10-09 JST, the portable CPU path was remeasured after the explicit
host-upload and weight-cache registration fixes at `6601a11` (eight Rayon
threads, two warmups, median of five iterations). Laya L64 B1 measured
192.8 ms and 209.8 ms on repeat; L8 B8 measured 182.0/182.8 ms. L8 B1 was
106.0/103.7 ms and L16 B1 was 108.4/110.0 ms. Reports, including the host
oracle and environment metadata, are in
[`bench-explicit-upload-cpu-2026-10-09`](../../../../fixtures/bench-explicit-upload-cpu-2026-10-09).
These runs show no observed slowdown relative to the historical portable
225.0/216.6 ms L64/batch report, but do not isolate the registration changes:
the host oracle also ran faster, and no contemporaneous old-revision baseline
was measured. The Python/Julia Laya gap remains unresolved.

### Laya extension execution profile (2026-10-09)

At `4b08835`, temporary `Instant` instrumentation measured the five CPU
extension families on the same production checkpoint, portable provider,
eight Rayon threads, two warmups and five measured forwards. The source was
restored after measurement. Each sample sums intervals within one family for
one forward; the table reports independent medians in milliseconds.

| shape | forward | GEMM | biased GEMM | LayerNorm | GeGLU | attention |
|---|---:|---:|---:|---:|---:|---:|
| L8 B1 | 106.4 | 66.9 | 4.9 | 1.1 | 1.7 | 0.9 |
| L16 B1 | 109.8 | 65.4 | 5.2 | 2.0 | 2.7 | 1.6 |
| L64 B1 | 200.2 | 117.9 | 9.2 | 8.1 | 9.1 | 14.0 |
| L8 B8 | 184.8 | 112.6 | 9.4 | 8.0 | 8.8 | 3.2 |

Projection execution remains the largest measured component. Family intervals
include input handling, output allocation and kernel execution, but exclude
final output Tensor construction and outer eager dispatch. Independent medians
cannot be subtracted to precisely attribute the remaining time. Timers and
stderr emission add overhead, so these are diagnostic results, not a new
uninstrumented speed comparison. Kernel sampling was unavailable because
`perf_event_paranoid=4`. Raw per-family samples, metadata and method are in
[`laya.json`](../../../../fixtures/bench-laya-extension-profile-2026-10-09/laya.json).

### Jeff native constant-cache ablation (2026-10-09)

`bench_jeff_native` measures the fully TensorNative DeltaNet path on the
production Jeff checkpoint with cached weights, float32, eight Rayon threads,
two warmups and five measured forwards per length. Model loading is excluded.
At `d6bea59`, sequential runs compared normal constant reuse with an ablation
that temporarily forced `prepare_native_constants` on every layer. The native
ops and weight cache were otherwise identical; the source was restored afterward.
This isolates cache preparation on the current code, not changes from an older
revision. All four runs produced identical logits at each benchmark length.

| length | cached, first / repeat | prepare every layer, first / repeat |
|---|---:|---:|
| L8 | 250.3 / 269.6 ms | 269.5 / 270.8 ms |
| L16 | 279.0 / 281.0 ms | 284.3 / 293.9 ms |
| L64 | 418.5 / 414.3 ms | 410.3 / 416.4 ms |

The L8 first-run improvement was not reproduced at the same magnitude; L64
reversed ordering. These results do not show a clear speedup across all shapes.
Sequential order and scheduler/thermal variation limit attribution. Removing
repeated constant construction remains relevant to the device-resident CUDA
path, but no CUDA speedup is established. CPU TensorNative L64 still takes
about twice the original Python latency reported above; Jeff's default
optimized host path remains faster.

The release production-reference test now executes two native requests with
one workspace/cache and checks both against the committed Julia logits.
Both had maximum absolute error `1.1444092e-5` at reference scale `10.445133`.
The checkpoint and reference were present, so this result exercised the model
rather than the test's absent-snapshot skip path. Raw samples, metadata,
ablation method and parity evidence are in
[`bench-jeff-native-constants-2026-10-09`](../../../../fixtures/bench-jeff-native-constants-2026-10-09).


### Laya attention improvement (2026-10-09)

Tiled library GEMM in the existing CPU attention extension reduced actual
checkpoint L64 B1 inference from 194.4 to 178.2 ms in alternating pairs;
a repeat reduced 191.4 to 179.0 ms (6.5–8.3%). Eight Rayon threads, float32,
two warmups and 15 pairs were used. Short sequences keep the same algorithm.
This remains slower than the original Python result of about 135 ms and Julia
result of about 122 ms, so issue #2 remains open. The actual Julia production
question/action parity test passed. See
[`22_CPU_KERNEL_OPTIMIZATION.md`](22_CPU_KERNEL_OPTIMIZATION.md) for selection
rules, numerical differences and replayable records. A prepared weight layout
experiment was rejected because warm gains were small or mixed and startup
became slower; its source prototype is retained only as a diagnostic artifact.


### Fresh Python speed target (2026-10-09)

With five warmups and 15 measured forwards, sequential processes and eight
threads, the original Python Laya measured 87.2 ms at L8 B1, 134.5 ms at L64 B1
and 134.6 ms at L8 B8. The production cached-tenferro baseline measured
99.5/185.6/183.4 ms respectively. This is a comparison with the production path;
the faster short-sequence host oracle is not used to claim a production win.
The fused MLP operation above reduces L64 to approximately 178.5–179.5 ms in
paired measurements, but the Python target remains unmet. Benchmark tooling
now records native raw samples and both marker/action outputs, and validates
all measured native shapes against the host oracle. Python records action
outputs as well as marker logits. See the new fixture directories linked in
`22_CPU_KERNEL_OPTIMIZATION.md` for source, accuracy and measurement limits.
