# tenferro-gated-delta Design

**Status:** implemented (CPU); CUDA pending hardware  
**Date:** 2026-10-04  
**Depends on:** `01_DESIGN.md` §4, `02_SPECIFICATION.md` §5, `04_MODEL_MAPPING.md` §2,
`09_JEFFCLIENT_ANALYSIS.md` §7, `11_TENFERRO_API_SURVEY.md`

> **Implementation status.** The host recurrent reference, the tenferro chunked
> formulation, the fused host recurrent kernel, prepared plans/workspaces
> (§9, §11), the direct `gated_delta` entry (§12), and the `GatedDelta`
> extension op (§12) are implemented and cross-checked in
> `crates/tenferro-gated-delta` (see `18_IMPLEMENTATION_STATUS.md`). CUDA (§10)
> remains unimplemented: no CUDA hardware is available here to validate it.

`tenferro-gated-delta` is the crate that extends tenferro-rs with the
Qwen3.5 Gated DeltaNet execution needed by `jeff-infer`. It owns the causal
depthwise convolution and both the reference and optimized Delta formulations.
It does **not** modify tenferro-rs; it depends on the published crates and adds
extension operations.

---

## 1. Scope and responsibilities

In scope:

- causal depthwise convolution (depthwise causal conv1d + SiLU)
- Q/K L2 normalization and Q scaling
- `beta`, `decay` precomputation helpers
- reference (recurrent) Delta formulation
- optimized CPU formulation (chunked triangular solve)
- specialized CUDA formulation (recurrent and/or chunked)
- deterministic algorithm selection
- prepared plans and reusable workspaces

Out of scope:

- checkpoint loading, config parsing, tensor naming (owned by `jeff-infer`)
- the surrounding Qwen3.5 layer, RMSNorm, MLP, readout (owned by `jeff-infer`
  and `tenferro-infer`)
- training, autodiff, generation, KV cache (package non-goals)

---

## 2. Relationship to tenferro-rs

- The crate depends on `tenferro-tensor` (storage, dtype, `BackendSession`),
  `tenferro-runtime` (extension machinery), `tenferro-cpu`, `tenferro-linalg`
  (for `triangular_solve`), and `tenferro-einsum` (contractions). CUDA support
  is behind an optional `cuda` feature depending on `tenferro-gpu`.
- It exposes operations through the extension API (`ExtensionOp` /
  `define_extension_runtime!`) so they can run in the direct, eager, and traced
  tiers, plus a convenience direct function.
- All hot operations use tenferro layouts (column-major) and explicit sessions;
  no hidden host/device transfer.
- Required tenferro primitives per `11_TENFERRO_API_SURVEY.md`: `dot_general`
  (batch dims), `reduce_sum`, `reduce_sum_squares`, `reduce_max`, `exp`, `sqrt`,
  `rsqrt`, `mul`/`add`/`sub`, `gather`/`slice`/`concatenate`, `clip`/`maximum`,
  and `triangular_solve(unit_diagonal = true)`. `cumsum`, softmax, and conv1d
  are provided by this crate (cumsum via a precomputed triangular-ones
  `dot_general` or a small fused kernel; conv1d as a dedicated extension op).

---

## 3. Crate layout and public API

```text
crates/tenferro-gated-delta/
├── src/
│   ├── lib.rs
│   ├── config.rs        # GatedDeltaConfig, Algorithm
│   ├── weights.rs       # GatedDeltaWeights (prepared tensor refs)
│   ├── plan.rs          # GatedDeltaPlan, shape buckets, selection
│   ├── workspace.rs     # GatedDeltaWorkspace (CPU/GPU scratch)
│   ├── conv.rs          # causal depthwise conv (+ SiLU)
│   ├── reference.rs     # recurrent host reference
│   ├── chunked.rs       # chunked CPU formulation
│   ├── recurrent.rs     # optimized recurrent CPU formulation
│   ├── extension.rs     # ExtensionOp wiring
│   └── cuda/            # feature = "cuda"
│       ├── mod.rs
│       ├── conv.rs
│       └── delta.rs
└── tests/
    ├── parity_reference.rs
    ├── chunk_boundaries.rs
    └── cuda_parity.rs   # feature = "cuda"
```

Public surface (proposed):

```rust
pub struct GatedDeltaConfig {
    pub key_heads: usize,          // linear_num_key_heads
    pub value_heads: usize,        // linear_num_value_heads
    pub key_dim: usize,            // linear_key_head_dim
    pub value_dim: usize,          // linear_value_head_dim
    pub hidden: usize,
    pub conv_taps: usize,
    pub eps: f32,                  // rms_norm_eps
    pub chunk_size: usize,         // default 64
    pub algorithm: AlgorithmChoice,
}

pub enum AlgorithmChoice { Auto, Reference, Recurrent, Chunked }

pub enum Algorithm { Reference, Recurrent, Chunked, CudaRecurrent, CudaChunked }

pub struct GatedDeltaWeights<'a> {
    pub qkv: TensorView<'a>,       // (hidden, 2*key_dim*key_heads + value_dim*value_heads)
    pub z: TensorView<'a>,         // (hidden, value_dim*value_heads)
    pub a: TensorView<'a>,         // (hidden, value_heads)
    pub b: TensorView<'a>,         // (hidden, value_heads)
    pub conv: TensorView<'a>,      // (conv_taps, conv_channels)
    pub a_decay: TensorView<'a>,   // (-exp(A_log)), (value_heads,)
    pub dt_bias: TensorView<'a>,   // (value_heads,)
    pub norm: TensorView<'a>,      // (value_dim,)
    pub out_proj: TensorView<'a>,  // (value_dim*value_heads, hidden)
}

pub struct GatedDeltaPlan { /* resolved algorithm, buckets, workspace layout */ }
pub struct GatedDeltaWorkspace { /* reusable buffers, one per concurrent context */ }

/// Convenience direct entry point.
pub fn gated_delta<'a>(
    session: &mut dyn BackendSession,
    x: &Tensor,          // (hidden, L)
    mask: &Tensor,       // (L,) f32, 0/1
    weights: &GatedDeltaWeights<'a>,
    plan: &GatedDeltaPlan,
    ws: &mut GatedDeltaWorkspace,
) -> Result<Tensor>;     // (hidden, L)
```

`resolve_algorithm(config, choice, backend_caps) -> Result<Algorithm>` is a pure function; the
resolved value is frozen into the plan and the extension-op identity.
`GatedDeltaPlan::resolve` also returns `Result`. Until CUDA execution is
implemented, a declared CUDA capability returns `UnsupportedConfig`, including
when CPU capability is also declared; no algorithm override silently selects
the host path. A backend with neither capability is rejected as well. Callers
choosing CPU explicitly use `BackendCaps::cpu()` or `cpu_simd()`.
`GatedDeltaPlan::from_config` remains the infallible convenience constructor
for an explicitly selected CPU formulation.

---

## 4. Algorithm selection

Requirements (`02_SPECIFICATION.md` §5.2): a reference path MUST exist; the
optimized path MAY choose recurrent or chunked; selection MUST be deterministic
for the same model/config/runtime.

Proposed policy:

The CUDA row is the target policy; current CUDA requests fail with the typed
error described above.

| Backend | `Auto` default | Override |
|---|---|---|
| Host reference (tests) | `Reference` | — |
| CPU portable | `Chunked` (chunk 64) | `Recurrent`, `Chunked`, `Reference` |
| CPU with `key_dim <= 256` and SIMD path | `Recurrent` | as above |
| CUDA | `CudaRecurrent` for `key_dim <= 256`, else `CudaChunked` | as above |

`Algorithm` is a semantic parameter of the extension op, so two ops with
different algorithms are not interchangeable. Determinism is guaranteed by
`resolve_algorithm` depending only on config, target, and declared capability —
never on runtime load or input values.

---

## 5. Weight contract

`jeff-infer` loads safetensors (see `09_JEFFCLIENT_ANALYSIS.md` §3) and converts
to the canonical runtime layout. `GatedDeltaWeights` receives prepared,
backend-resident tensor views:

- `a_decay = -exp(A_log)` is computed once at load time, not per request.
- `qkv` is a single concatenated projection; the `q/k/v` column ranges are
  `key_dim*key_heads`, `key_dim*key_heads`, `value_dim*value_heads`.
- `conv` is the depthwise kernel with `conv_taps` taps over
  `2*key_dim*key_heads + value_dim*value_heads` channels.
- All weights are `f32`; the layer validates shapes at plan construction.

The crate does not perform checkpoint parsing or tensor-name lookup.

---

## 6. Reference formulation (recurrent)

Inputs: normalized hidden `x` `(hidden, L)`, mask `(L,)`.
Precompute (per `09_JEFFCLIENT_ANALYSIS.md` §7.1):

```text
masked = x * mask
mixed  = SiLU(causal_depthwise(qkv^T masked))
q = L2norm(mixed[0 : key_width]) / sqrt(key_dim)      # key_width = key_dim*key_heads
k = L2norm(mixed[key_width : 2*key_width])
v = mixed[2*key_width : end]                          # value_dim*value_heads
z = SiLU(z^T masked)                                  # value gate
beta  = sigmoid(b^T masked)                           # (value_heads, L)
decay = a_decay * softplus(a^T masked + dt_bias)      # (value_heads, L)
```

For each value head `h` with `kh = ceil(h / (value_heads / key_heads))` and
state `S` of shape `(value_dim, key_dim)`, initialized to zero:

```text
for t in 0..L:
    factor   = exp(decay[h, t])
    pred     = S @ k[:, kh, t]
    corr     = beta[h, t] * (v[:, h, t] - factor * pred)
    S        = factor * S + outer(corr, k[:, kh, t])
    result   = S @ q[:, kh, t]
    out[:, h, t] = rms_noncentered(result, norm) * z[:, h, t]
out = out_proj^T @ out_reshaped
```

This is the correctness reference. It is implemented as pure host loops over
`Tensor` host slices so that it shares no optimized kernel with the production
path (`05_TESTING_BENCHMARKS.md` §2).

---

## 7. Chunked formulation

Chunk size `C = config.chunk_size` (default 64). For each value head and each
chunk `[start, end)`:

Let `c_i = sum_{p<=i} decay[p]` (cumulative decay), `weighted_i = beta_i k_i`.

Define the effective linear system

```text
M = I + L
L[i, j] = beta_i * (k_i · k_j) * exp(c_i - c_j)   for i > j
L[i, j] = 0                                        otherwise
```

`M` is unit lower triangular. The raw matrix computed as
`(weighted^T k) .* pair_decay`, where `pair_decay[i,j] = i>=j ? exp(c_i-c_j) : 0`,
has diagonal `beta_i * ||k_i||^2 ~= beta_i`; that diagonal is **intentionally
ignored** by the unit-triangular solve. This matches the reference Julia
implementation (`09_JEFFCLIENT_ANALYSIS.md` §7.3) and is a required detail.

Per chunk:

```text
rhs_values[i, :] = beta_i * v_i                       # (n, value_dim)
rhs_keys[i, :]   = beta_i * exp(c_i) * k_i            # (n, key_dim)
X = M^{-1} rhs_values                                 # triangular solve
Y = M^{-1} rhs_keys                                   # triangular solve
corrections = X^T - S @ Y^T                           # (value_dim, n)
intra[i, j]  = (k_i · q_j) * exp(c_j - c_i)  (j >= i) # (n, n)
result = S @ (q .* exp(c)) + corrections @ intra      # (value_dim, n)
ending_keys[:, j] = exp(c_end - c_j) * k_j
S = corrections @ ending_keys^T + exp(c_end) * S
out[:, span] = rms_noncentered(result, norm) * z[:, span]
```

Both triangular solves are `triangular_solve(M, rhs, left_side=true,
lower=true, transpose_a=false, unit_diagonal=true)`; `intra` and the cumulative
sums use contractions and elementwise ops. The final output head layout is
`(value_dim, value_heads, L)`, reshaped to `(value_dim*value_heads, L)` for
`out_proj`.

Chunk size `C` is part of the plan (and op identity) because it changes the
number and shapes of the intermediate systems. `C` must divide or tile `L`
exactly; the last chunk may be shorter.

---

## 8. Causal depthwise convolution

```text
out[ch, t] = sum_{tap=0..taps-1} in[ch, t - (taps-1-tap)] * w[tap, ch]   if t-(taps-1-tap) >= 0
out = SiLU(out)   # fused
```

- Depthwise: one weight row per channel; no cross-channel mixing.
- Causal: no future taps; no padding.
- Masked positions are zeroed before the projection, so they contribute zero.
- SiLU is fused into the convolution (as the Julia reference does).

Reference: composed from `slice`/`pad`/`mul`/`add` shifts for tap parity tests.
Optimized CPU: a dedicated loop kernel. CUDA: `cuda::raw` kernel.

---

## 9. CPU execution design

- **Reference** (`Algorithm::Reference`): host loops, no tenferro ops.
- **Chunked** (`Algorithm::Chunked`): tenferro `dot_general` /
  `triangular_solve` / reductions. This is the portable production path and
  matches the Julia CPU default.
- **Recurrent** (`Algorithm::Recurrent`): a fused CPU loop over tokens with the
  per-head state held in a workspace buffer; used where it benchmarks faster
  (e.g. `key_dim <= 256`). It may be implemented as a Rust loop over compact
  host slices rather than a graph.

Shared CPU concerns:

- Q/K normalization is computed once per key head (Q/K heads are shared by
  multiple value heads) and reused, as in the reference.
- `decay`, `beta`, and `z` activations are precomputed once per layer.
- Workspace buffers are sized at plan time and reused; no per-request large
  allocation (target, not a first-prototype requirement).

---

## 10. CUDA execution design

Two CUDA formulations mirror the reference implementations:

1. **Recurrent kernel**: one warp (or a small thread group) owns one value row;
   the `(value_dim, key_dim)` state is held in registers/shared memory across
   the token scan, expressing `S = factor*S + outer(corr, k)` and `result = S@q`
   in one pass. Launch and reduce the gate/normalization in fused kernels.
   Preferred for `key_dim <= 256`.
2. **Chunked kernel**: fused convolution + Q/K normalize + per-chunk
   contraction and a unit-triangular solve. Used for `key_dim > 256`, where the
   Julia Metal backend also keeps the chunked solve as the stable fallback.

Constraints:

- Weights, constants, and workspaces stay device-resident after load; per
  request only compact input upload and output download cross the boundary.
- Model weights are device-resident; `GatedDeltaWorkspace` is backend-owned and
  must not be reused while queued work may still read it (stream ordering).
- No silent CPU fallback: unsupported dtype/shape returns a typed error.
- The CUDA implementation depends on the same public `cuda::raw` boundary
  documented in `11_TENFERRO_API_SURVEY.md` §9.

CUDA support is behind a `cuda` feature; the crate compiles and its CPU paths
remain testable without a GPU. The feature also enables the CUDA dispatch
in `tenferro-ad` and `tenferro-linalg`, required for a native triangular-solve
fallback; enabling only `tenferro-gpu/cuda` does not enable that linalg route.

The current optional component is `cuda::CudaKernels`, which compiles and holds
raw handles for convolution/SiLU, the register-state scan (`key_dim <= 256`)
with fused Q/K normalization and beta/decay gates, and output RMSNorm/gating.
It is not yet wired into full layer dispatch. The scan owns one warp per value
row; the norm/gate epilogue is separate because it reduces across value rows.
The source documents buffer layouts, argument order, and launch geometry.
Loaded modules must stay in a thread-bound execution owner until queued work
completes; the raw launch contract is not automatically managed by this helper.

The `cuda::RecurrentGeometry` adapter validates nonzero dimensions, grouped
head divisibility, the register-state key limit, CUDA grid-y limits, and all
signed 32-bit indexing products before exposing scalar arguments and launch
geometry. GPU-free tests cover token boundaries and overflowing dimensions.
`CudaKernels::enqueue_stages` now submits the three stages without a host
barrier. It validates exact f32 tensor shapes, zero-offset column-major
layout, runtime identity, device residency, allocation spans, and positive
finite RMS epsilon before any launch. The method remains unsafe: the caller
must ensure disjoint underlying allocations and retain all buffers/module
until successful stream synchronization, including when a later launch
fails. A safe full-layer execution owner and hardware parity remain pending.

The ignored hardware test `tests/cuda_stages.rs` compares the three-stage
output with causal convolution plus the existing CPU recurrent oracle for
lengths 1/63/64/65/127/128/129, key widths 1/33/256, and grouped heads. It
synchronizes once after all stages, including after enqueue errors; if the
barrier fails it intentionally retains the allocations and kernel module.
This is a raw-stage gate, not full-layer mask/padding or large-key coverage.
Run it explicitly on a CUDA runner with NVRTC available:

```sh
TENFERRO_CUDA_ARCH=compute_80 cargo test -p tenferro-gated-delta --features cuda --test cuda_stages -- --ignored
```

A second ignored gate, `native_unit_lower_triangular_solve_supports_large_key_rhs`,
checks native CUDA linalg with triangular widths 1/63/64/65 and 257 RHS
columns. Non-unit diagonal and large upper entries verify that the lower/unit
flags are respected; the result must stay device-backed before download.
This checks the library building block only. Device-side cumulative decay
preparation is implemented below;
The contraction/solve step is implemented below; CUDA chunk-loop integration
and full large-key DeltaNet parity remain pending.

Select an architecture supported by the device. On 2026-10-08 this explicit
run failed before launch because the environment lacks `libcuda`; the test
compiled, but hardware numerical parity is still unproven.

`CudaKernels::enqueue_chunk_decay` now prepares chunk gates and decay factors
on device: cumulative **log** decay, beta, lower-triangular pair weights,
ending-key tail weights and the final chunk decay. One block owns a value-head
chunk; upper-triangular and padded pair entries are explicitly zeroed. The
log-prefix storage preserves finite pair/tail differences when the final
factor underflows. `ChunkDecayGeometry` bounds chunk width to 1..=256 and
checks signed 32-bit indexing products. The unsafe enqueue adapter validates
all f32 shapes/layouts, runtime/device residency and allocation spans; buffer
aliasing and lifetime management remain the caller's obligations.

The ignored `chunk_decay_matches_cpu_with_padding_and_underflow` hardware gate
covers lengths 1/63/64/65/127/128/129 and strongly negative decay. It compares
all five outputs, including pair padding and tail weights with a zero final
factor. This gate compiles but has not run on a CUDA device. The compiler
report for all four exported kernels is
`fixtures/cuda-compile-2026-10-08/nvrtc-chunk-decay.json`; the earlier report
records the original three-stage source hash. Device preparation and the native chunk step are implemented separately;
CUDA chunk-loop integration and full-layer dispatch remain pending.

The backend-agnostic `chunked::delta_scan_chunk_step` now accepts prepared
`EagerTensor` inputs and returns the next state and gated chunk output. It
composes contractions, two unit-lower `triangular_solve` calls, corrections,
state update, RMSNorm and SiLU entirely through session ops. Epsilon,
inverse value width and one are prepared scalar tensors supplied by the
caller; the step creates no host constants, reads no tensor values onto the
host and adds no explicit synchronization. Tensor shapes are validated before
computing. No backend fallback is introduced.

The existing host-input `delta_scan_chunked` wrapper now calls this same
step; it still prepares host decay factors and downloads chunk outputs for
its `Vec<f32>` API. CPU oracle tests cover key width 257 at lengths 1/63/64/65,
as well as the existing unequal key/value widths and strong-decay boundary
regressions. They now reject non-finite outputs explicitly. A malformed norm
weight is checked to return a typed error. This verifies the contraction/solve
building block on CPU, not device preparation plus a complete CUDA scan.
The CUDA adapter still needs to bind prepared device factors to eager tensors,
iterate heads/chunks with stream-ordered ownership and assemble device outputs.

For compiler validation without a GPU, run:

```sh
python3 tools/check_cuda_kernels.py --nvrtc-library "$NVRTC_LIBRARY"
```

This validates PTX generation and parameter widths/order for virtual targets
`compute_70`, `compute_80`, and `compute_90`. It neither launches kernels nor
proves CPU/CUDA numerical parity, residency, or asynchronous lifetime safety.
Those exit criteria, large-key chunked execution, and the full layer adapter
remain open. CUDA plan requests continue to return `UnsupportedConfig`.
The 2026-10-08 compiler report is saved in
`fixtures/cuda-compile-2026-10-08/nvrtc.json`, using the NVIDIA
`nvidia-cuda-nvrtc-cu12==12.6.85` distribution (NVRTC version 12.6).

---

## 11. Workspace and prepared plans

```rust
pub struct GatedDeltaPlan {
    algorithm: Algorithm,
    chunk_size: usize,
    // shape-specialized buffer layout for known sequence lengths
}

pub struct GatedDeltaWorkspace {
    // per key head: q, k, v, z, beta, decay
    // per chunk: system, rhs buffers, corrections, intra, ending keys
    // per value head: recurrent state
}
```

- Plans are shape-bucketed for common lengths (64/128/256/512, aligned with
  `01_DESIGN.md` §8) with a generic fallback for correctness.
- A workspace is owned by an inference context, not global; concurrent requests
  use separate workspaces (`02_SPECIFICATION.md` §14).
- GPU workspaces respect asynchronous execution lifetimes.

---

## 12. Extension-op integration

For traced/eager integration, expose a `GatedDelta` extension operation via
`tenferro_runtime::extension` and `define_extension_runtime!`:

- Descriptor fields are semantic only: `key_heads`, `value_heads`, `key_dim`,
  `value_dim`, `conv_taps`, `eps` bits, `chunk_size`, `algorithm`, and the
  normalization mode. `x`, `mask`, and weights are inputs.
- `execute_reads` receives borrowed `TensorRead` inputs and the
  `ExtensionExecutionContext`; it dispatches to the CPU or CUDA implementation.
- No hidden process-global cache; any retained plan is owned by the runtime or
  eager context and bounded.
- The convenience direct function is the primary API for `jeff-infer`; the
  extension op exists for graph reuse and uniformity.

The causal depthwise convolution may be a second extension op
(`CausalDepthwiseConv1d`) or fused into `GatedDelta`; profiling decides.

---

## 13. Numerical policy

- MVP dtype `f32`; accumulate in `f32`.
- The recurrent and chunked formulations must agree within the documented
  tolerance; the reference must agree with the Julia/PyTorch references within
  the same tolerance used by `05_TESTING_BENCHMARKS.md`.
- `exp`/`softplus`/`sigmoid` must follow the reference definitions
  (`softplus(x) = max(x,0) + log1p(exp(-|x|))`) to keep parity at extremes.
- `rms_noncentered` uses `eps` inside the square root and no `1 + w` term.
- Q uses `eps = 1e-6` for L2 normalization and scales by `1 / sqrt(key_dim)`
  (the reference divides; the sign of this factor only shows through the
  output-RMSNorm `eps`, but it is load-bearing for fixture parity); K uses
  `eps = 1e-6` with no scale.
- Algorithm choice MUST NOT change semantic output beyond tolerance; any
  divergence is a bug, not an acceptable approximation.

---

## 14. Testing and parity

- **Cross-formulation parity**: reference vs recurrent vs chunked on the same
  inputs, at f32 tolerance.
- **Intermediate activations** (per `05_TESTING_BENCHMARKS.md` §3): convolution,
  q/k/v, beta, decay, chunk solve results, per-chunk output, final layer output.
- **Boundary shapes**: `L = 1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 512`
  (chunk boundaries), plus mask holes and left padding, and `value_heads`
  multiples of `key_heads`.
- **Determinism**: identical config/backend yields identical `Algorithm` and
  bit-stable selection; no input-dependent branch.
- **CUDA parity**: CPU vs CUDA for the regression suite; device memory released
  between runs; scalar GPU indexing disabled as needed.
- **Reference independence**: the host reference shares no kernel with the
  optimized paths.

---

## 15. Benchmark and profiling hooks

- Report algorithm, chunk size, dtype, sequence length, batch, backend, and
  whether conv/normalization are fused.
- Separate the convolution, projection, chunk/recurrent core, and output
  projection timings.
- CUDA: report launch count, DRAM traffic, and whether state is retained
  on-chip, per `03_CPU_GPU_INFERENCE.md` §3.6.
- CPU: SIMD/reduction behavior for the recurrent loop; GEMM provider for
  contractions.

---

## 16. Open questions

- [ ] Does the chunked path benefit from a dedicated fused extension op, or is
      the composed `dot_general`/`triangular_solve` fast enough on CPU?
- [ ] Which CUDA granularity pays off for the checkpoint's `value_dim` and
      `key_dim` (warp-per-row vs block-per-head)?
- [ ] Exact handling of `key_dim > 256` on CUDA: native chunked kernel vs
      library `triangular_solve`.
- [ ] Whether `cumsum` should be a fused extension op or the triangular-ones
      contraction is sufficient.
- [ ] Batched (multi-sequence) Delta execution: per-sample state layout and
      whether the extension op should accept a batch dimension.
- [ ] Tolerance values to publish for CPU and CUDA.
- [ ] Whether convolution should be exposed as its own extension op for reuse.
