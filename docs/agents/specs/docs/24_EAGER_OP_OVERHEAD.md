# Eager vs Compiled Execution

**Status:** analysis complete; compiled prototype in progress
**Date:** 2026-10-04
**See:** `21_SPEED_COMPARISON.md`, `23_TENFERRO_NATIVE.md`, `AGENTS.md`

## Question

Why is the tenferro-native CPU forward slower than the bespoke host path, and
can the per-op eager overhead be fused?

## "host" vs "tenferro"

- **host** = the CPU host-slice path (`cpu-kernels`: `matrixmultiply` + `rayon`),
  tenferro-free. It is the correctness oracle and the CPU-competitive path.
- **tenferro** = the `EagerSession` op path.

Both run on the same CPU; the distinction is implementation style, not device.

## Eager execution cost (from the source)

- Every eager op goes through `EagerTensor::nary_op_in_session`
  (`tenferro-ad/src/eager_ops.rs:214`): context check → collect input
  `TensorRead`s → execute the op on the backend → allocate the output → wrap the
  result. There is **no fusion and no graph**.
- Inference takes the `!any_requires_grad` fast branch: with constant inputs
  `eager_grad_recording_enabled()` is true but no tensor `requires_grad`s, so
  **no autograd nodes are recorded**. AD is *not* the cost.

## Profiler

Enable with (see `tenferro-ad/src/eager.rs:322-390`):

```sh
TENFERRO_PROFILE_EAGER_OP_AGG=1 TENFERRO_PROFILE_EAGER_OP_PRINT_EVERY=<N> <cmd>
```

It prints aggregated per-section timings every `N` `nary_op` calls.

Measured on the production Jeff checkpoint (`RAYON_NUM_THREADS=8`, all eager
forwards in one bench run):

| section | share of `nary_op.total` | per call |
|---|---:|---:|
| `nary_op.total` | 100% (1770.6 ms, 14000 ops) | 126.5 µs |
| **`nary_op.exec_single_output_read`** | **98.3%** | 124.3 µs |
| `collect_input_reads` | 0.7% | 0.90 µs |
| `new_untracked_result` | 0.4% | 0.49 µs |
| `requires_grad_scan` | 0.04% | 0.05 µs |
| `context_check` | 0.03% | 0.04 µs |

**Reading:** the Rust-side bookkeeping is **under 1.5%**. **98.3% is the
per-op backend execution itself** (`exec_single_output_read`). The lever is
therefore to **call the backend fewer times** — fuse consecutive elementwise
ops into one kernel and stop materializing intermediates — not to trim the
harness.

## tenferro's fusion lives in the compiled path

The eager API does not fuse by design, but tenferro ships a separate
**compiled (lazy) path** that does:

- `GraphCompiler::compile(&TracedTensor) -> CompiledGraph`
  (`tenferro-runtime/src/graph/compiler.rs:172`). `OptimizerConfig` defaults to
  `algebraic_layout_simplifier = true`, `dot_decomposer = false`
  (`compiler/options.rs`).
- `segment_exec_program` groups instructions into `Segment::Fused`;
  `build_elementwise_fusion_plan` (`segment.rs:1452`) turns a consecutive
  elementwise chain into an `ElementwiseFusionPlan`, executed by
  `session.execute_elementwise_fusion` (`segment.rs:850, 901`) →
  `tenferro-cpu-fused` → `strided_fused` (SIMD fused elementwise).
- `GroupedGemmConfig` (`tenferro-tensor/src/backend.rs:469`) for grouped GEMM.
- Run it with `Runtime::builder()
  .register_engine(tenferro_cpu::runtime_engine_registration(&backend))` then
  `runtime.run_compiled(&program, &inputs)` (`runtime/snapshot.rs:1004`).
- Extension ops can also be run as a one-op compiled program:
  `SemanticProgramBuilder` + `GraphCompiler::compile_frozen_program` +
  `run_compiled` (`tenferro-ad/src/eager_exec.rs:255`).

`TracedTensor` covers the ops the forward needs, including linalg through
`tenferro_linalg::TracedTensorLinalgExt` (`triangular_solve`, `solve`, …).

## Conventions to reuse (confirmed against both paths)

- `dot_general` emits **batch dimensions trailing** (`TracedTensor::dot_general`
  shape hint: free-lhs, free-rhs, batch). Fold back to batch-leading.
- `reshape` preserves **column-major** order, so a `[width, L]` head split is
  `reshape (dim, heads, L)` then transpose (as the attention code already does).
- `triangular_solve` is **rank-2 only**, so the per-chunk solves loop over heads
  while the rest of the scan stays batched.

## Implications

- Keep **eager** where flexibility matters and as the reference.
- Move the hot forward to the **compiled** path: express it on `TracedTensor`,
  compile once per shape, cache the `CompiledGraph`, and run through `Runtime`.
  Expected effect: elementwise chains (norms, gates, `softplus`/`silu`, conv
  taps, softmax) fuse into SIMD kernels with no intermediates; GEMM segments
  stay separate (unless grouped).
- In progress: `tenferro-gated-delta::traced_layer` — the head-batched DeltaNet
  re-expressed on `TracedTensor` — plus a benchmark comparing it against the
  eager native layer.

## Prototype: compiled DeltaNet layer

`tenferro-gated-delta::traced_layer` re-expresses the head-batched DeltaNet on
`TracedTensor`; `examples/bench_compiled_delta.rs` compiles it once and runs it
via `prepare_compiled` + `run_prepared` (the steady-state API — `run_compiled`
re-prepares every call). Measured at `hidden=1024, length=64, value_heads=16`
(release, 8 threads):

| benchmark | eager | compiled | compiled / eager |
|---|---:|---:|---:|
| DeltaNet layer | 21.7–27.9 ms | 18.9–24.0 ms | **~1.1×** |
| 200-op elementwise `tanh` chain, 1024×64 | 23–35 ms | 71–128 ms | **0.27–0.33×** |

The compiled layer matches the eager layer to `1.6e-5`, and saves ~10% on the
DeltaNet layer. The pure elementwise chain, however, is **3–4× slower** on the
compiled path — the opposite of the expected fused-SIMD win. So on these graphs
fusion is either not engaging (the segment executor appears to fall back to
per-instruction staging) or its per-instruction cost dominates at these sizes.

**Takeaway:** the DeltaNet layer is GEMM/solve-bound, so elementwise fusion
cannot move it much; and the compiled elementwise path needs investigation
before it can be adopted. Confirming whether the CPU runtime actually executes
via `Segment::Fused` + `execute_elementwise_fusion` for these graphs (there is
no public segment dump; a targeted microbench or upstream question is needed) is
the next step.

## Reference: eager native trajectory (Jeff, L8)

| stage | time |
|---|---:|
| host + GEMM + rayon (oracle) | 177 ms |
| tenferro chunked, host round-trips | 1693 ms |
| tensor-native, head-sequential | 496 ms |
| tensor-native, head-batched | 418 ms |
| + merged solves, attention in-session | 342 ms |
