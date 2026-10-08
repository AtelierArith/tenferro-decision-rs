# tenferro-rs API Survey for the Decision Engines

**Status:** analysis / implementation reference  
**Date:** 2026-10-04  
**Surveyed revision:** `471c4278` (workspace version `0.7.1`)  
**Note:** the package's `08_SOURCE_SNAPSHOT.md` recorded `67baa74b`, which is now
an ancestor of this revision. Re-validate against this revision before coding.

This survey maps the operations Laya and Jeff need onto the current tenferro-rs
public API, records the gaps, and identifies the extension and CUDA hooks the
package's `tenferro-infer` / `tenferro-gated-delta` crates will use.

---

## 1. Crate selection

There is no `tenferro` facade crate. A CPU concrete program depends on:

```toml
tenferro-runtime = { version = "0.7", features = ["cpu-faer"] }
tenferro-cpu     = { version = "0.7", features = ["cpu-faer"] }
```

Add `tenferro-einsum`, `tenferro-linalg`, or `tenferro-fft` for those operation
families. Add `tenferro-gpu` with the `cuda` feature for CUDA (and the
experimental `webgpu` feature for WebGPU). At least one of `cpu-faer` /
`cpu-blas` is required; enable exactly one BLAS provider when using `cpu-blas`.

---

## 2. Execution model

- **Direct concrete tensors** run inside an explicit session:
  ```rust
  let y = backend.with_backend_session(|session| x.matmul(&w, session))??;
  ```
  Session entry is fallible and wraps the callback's own `Result`; operations
  returning `Result` therefore need `??`. Re-entering the same backend from
  inside a session fails with `Reentered`.
- **Amortize entry** with `CpuBackend::with_execution_scope` for sequential
  ordinary/eager/prepared work built from clones of one configured backend
  (CPU domains only).
- **Eager** uses `EagerRuntime::with_eager_session(|s| …)??`; family methods
  come from `EagerSessionEinsumExt` / `EagerSessionLinalgExt`.
- **Traced** compiles once (`GraphCompiler`) and replays (`Runtime::run_compiled`);
  extension modules must be installed and the engine registered.
- **Prepared contractions**: `ConcreteEinsumPlan` (and `execute_into` /
  `execute_read_into` for a validated preallocated destination) when equation,
  operand order/count, dtypes, and shapes repeat.

For the decision engines this means: construct the backend once at load time,
prefer one session for a request, and keep shape-specialized plans in the model.

---

## 3. Storage, dtype, layout

- Owned tensors are **column-major**; `from_vec_col_major` silently reinterprets
  row-major data (wrong values, no error). The safetensors loader must reorder.
- Public `Tensor` dtypes: `F32`, `F64`, `I32`, `I64`, `Bool`, `C32`, `C64`.
  **`F16`/`BF16` are not public dtypes yet**, so Phase 8 reduced precision is
  currently tenferro-blocked; the MVP `f32` path is unaffected (safetensors
  converts to `F32`).
- Views carry arbitrary strides; compact outputs are column-major. Layout
  conversion within a device is allowed; cross-device transfer is never implicit.

---

## 4. Operation coverage relevant to the engines

### Contraction

- `dot_general` with `lhs_contracting_dims`, `rhs_contracting_dims`,
  `lhs_batch_dims`, `rhs_batch_dims`. Output is
  `[lhs_free…, rhs_free…, batch…]` (**batch-trailing**).
- `matmul` (rank-2), einsum (`"ij,jk->ik"` arrow required), eager `tensordot`.

### Elementwise and analytic

`add, sub, mul, div, pow, maximum, minimum, neg, abs, sign, conj, exp, log,
sin, cos, tanh, sqrt, rsqrt, expm1, log1p`, plus `clamp`, `compare` (→`Bool`),
and `where_select` / `select`.

### Reductions

`reduce_sum`, `reduce_prod`, `reduce_max`, `reduce_min`, and
`reduce_sum_squares`.

### Structural / indexing

`reshape, transpose, broadcast_in_dim, convert / cast, extract_diagonal,
embed_diagonal, tril, triu, slice, dynamic_slice, pad, concatenate, reverse,
gather, scatter, to_contiguous_read, copy_read_into`, plus shape-packing
helpers `stack` and `index_select`.

### Linear algebra (`tenferro-linalg`)

`svd, qr, cholesky, solve, triangular_solve, lu, full_piv_lu, eig, eigh, pinv,
det, slogdet, norm`.

`triangular_solve(a, b, left_side, lower, transpose_a, unit_diagonal)` carries
the `unit_diagonal` flag needed by the chunked Gated DeltaNet solve.

### Explicitly absent

- `cumsum` / `cumprod`
- `softmax` / `log_softmax`
- `argmax` / `argmin`, `reduce_mean` / `reduce_var`
- `sigmoid` / `silu` / `softplus` / `gelu`
- `conv1d` / depthwise convolution
- `outer`

These are composed from existing ops or added as extension operations
(section 7).

---

## 5. Laya mapping

| Laya need | tenferro plan | Status |
|---|---|---|
| embedding gather | `gather` / `index_select` | ✅ |
| Linear / prepared GEMM | `dot_general` / `matmul` / `ConcreteEinsumPlan` | ✅ |
| LayerNorm | `reduce_sum` + `reduce_sum_squares` + `rsqrt` + broadcast `mul`/`add`/`sub` | △ compose |
| GELU / GeGLU | `tanh`/`exp` based, fused later | △ compose |
| softmax / masked softmax | `reduce_max`+`sub`+`exp`+`reduce_sum`+`div`, mask via `maximum`/`where_select` | △ compose |
| RoPE | precomputed cos/sin + `slice`/`reshape`/`concatenate` arithmetic | ✅ compose |
| attention | `dot_general` with head batch dims + composed softmax | ✅ compose |
| marker gather | `gather` / `index_select` | ✅ |
| decision head / scorer | `dot_general` + composed activations | ✅ |
| argmax / calibration | not in tenferro; implement in `laya-infer` | ℹ host |
| fused LayerNorm / attention / GeGLU | extension op, guided by profiling | planned |

## 6. Jeff mapping

| Jeff need | tenferro plan | Status |
|---|---|---|
| embedding gather | `gather` / `index_select` | ✅ |
| projections | `dot_general` | ✅ |
| RMSNorm (centered and not) | `reduce_sum_squares` + `rsqrt` + broadcast; centered adds ones | ✅ compose |
| partial RoPE | precomputed tables + structural ops | ✅ compose |
| GQA / grouped attention | `dot_general` batch dims (head groups) | ✅ |
| masked softmax | composed, `-floatmax` via `maximum`/`where_select` | △ compose |
| SiLU gate / MLP | `exp`+`mul` composed | △ compose |
| depthwise causal conv1d | extension op (or shift+`slice`+`concatenate` compose for a reference) | ❌ extension |
| Gated DeltaNet chunked | `triangular_solve(lower=true, unit_diagonal=true)` + `cumsum` | △ cumsum gap |
| Gated DeltaNet recurrent | composed elementwise/reduction loop (reference) or CUDA extension | ✅ reference |
| readout | `dot_general` + host calibration | ✅ |

---

## 7. Closing the gaps

- **sigmoid**: `1/(1+exp(-x))` or `0.5*(1+tanh(x/2))`.
- **SiLU**: `x * sigmoid(x)`.
- **softplus**: stable `log1p(exp(x))`, or `max(x,0)+log1p(exp(-|x|))`.
- **softmax**: `reduce_max` → `sub` → `exp` → `reduce_sum` → `div`.
- **cumsum**: multiply by a precomputed lower-triangular ones matrix via
  `dot_general`; or an extension op for the chunk-sized cumulative sums.
- **conv1d (depthwise, causal)**: extension op. A reference can be composed
  from shifted `slice` + `concatenate` product/sum for small tap counts, but the
  optimized path should be an extension or fused CUDA kernel.
- **argmax / calibration / confidence**: pure host logic in the engine crate.
- **FP16/BF16**: not available in the public dtype set; deferred with tenferro.

Fused primitives (LayerNorm/RMSNorm, masked softmax, GeGLU/SiLU, partial RoPE)
should become extension operations only where profiling proves the composed
form costs meaningful launch or memory traffic.

---

## 8. Extension operation mechanism

- Implement `tenferro_runtime::extension::ExtensionOp` for the operation
  descriptor (semantic parameters only; no hidden mutable caches).
- `define_extension_runtime!` generates the private engine / prepared adapter
  and an `extension_module` constructor. The required callback is
  `execute_reads`, receiving borrowed `TensorRead` inputs and an
  `ExtensionExecutionContext` over the backend session.
- `TensorRead::as_slice::<T>()` borrows compact host storage and returns a typed
  error for non-compact views or backend-owned buffers — no silent download.
  Canonicalize with `to_contiguous_read` explicitly.
- Extension state is owned by `EagerRuntime` / `GraphCompiler` / `Runtime`; no
  process-global mutable state. This matches the package's ownership rule
  (`01_DESIGN.md` §3.5).
- AD rules are optional; without them, AD reports the op unsupported rather than
  dropping gradients (irrelevant for inference-only crates).

This is the intended home for `tenferro-gated-delta` and for fused
`tenferro-infer` kernels.

---

## 9. Custom CUDA kernels

`tenferro_gpu::cuda::raw` is a public extension boundary:

```text
CudaBackend::with_backend_session
  -> with_cuda_exec_session
  -> CudaExecSession::with_raw
  -> load PTX/CUBIN or compile with NVRTC
  -> launch on the session stream
```

- `session.tensor()/tensor_mut()` validate residency and return bounded device
  spans; `alloc_output::<T>()` allocates; `raw::KernelArg` / `LaunchConfig`
  describe the launch.
- `NvrtcOptions`, `load_ptx`, `load_cubin`, `compile_nvrtc`. A raw module is
  `!Send`/`!Sync`, whereas `raw::Session::resource` requires `Send`; keep loaded
  modules in the thread-bound CUDA execution owner. The resource cache can
  retain compatible `Send` state, not a raw `Module` directly.
- Device-side libraries (cuBLASDx, cuSOLVERDx) are callable from downstream
  kernels.
- Stream ordering and lifetime rules: launch on the session stream, keep
  modules/inputs/outputs alive, synchronize only when the host must wait.

This supports bespoke fused DeltaNet, attention, and convolution kernels while
keeping tenferro's context, stream, and allocation ownership.

---

## 10. GPU coverage

**CUDA** (CubeCL-CUDA + cuTENSOR/cuBLAS/cuSOLVER) is broad for `F32`/`F64`/etc.:
elementwise and analytic ops, `reduce_sum`, `reduce_sum_squares`, `reduce_max`,
`dot_general`, `gather`/`slice`/`dynamic_slice`/`concatenate`/`reverse`, and
linalg including `triangular_solve`. Unsupported op/dtype returns
`BackendFailure` — never a silent CPU fallback.

**WebGPU/Metal** is intentionally narrow: currently `dot_general` for `F32`/`C32`
and transpose/to-contiguous. Elementwise, reductions, indexing, and linalg are
not implemented and return errors. Running Jeff or Laya on Apple GPU therefore
requires implementing most primitives as WebGPU extension operations. This
matches the package's decision to treat Apple GPU as a post-CPU/CUDA target, and
is a stronger constraint than `01_DESIGN.md` §11 implies.

CUDA library floor: CUDA 12.6.2 minimum, 12.8 full capability; cuTENSOR 2.1.x on
compute capability 7.0.

---

## 11. Performance and ownership idioms

- Construct the backend/runtime once; per-call construction discards pools and
  caches.
- Reuse one session for a request; use `with_execution_scope` where entry cost
  matters. Measured CPU session entry was ~26 us at 18 workers on one machine
  (machine-specific), so grouping calls matters for tiny operations.
- Keep `ConcreteEinsumPlan` / compiled programs in the model; the plan retains
  metadata only, and execution still validates and allocates.
- Use `execute_into` / `copy_read_into` for preallocated destinations where the
  package's workspace requirements demand no per-request allocation.
- CUDA: explicit upload/download; no hidden host round trip; synchronize only at
  boundaries (`EagerRuntime::synchronize` or `backend.runtime().synchronize()`).

---

## 12. Implications for the Rust design

1. **f32 MVP is compatible** with tenferro's dtype set; FP16/BF16 must wait for
   tenferro support.
2. **`triangular_solve` with `unit_diagonal`** makes the chunked Gated DeltaNet
   core a standard-op implementation; keep the recurrent form as the reference.
3. **`cumsum` and `conv1d` are the two real custom-operation candidates**;
   `softmax`, `sigmoid`, `SiLU`, and `softplus` are cheap compositions.
4. **`tenferro-gated-delta` should be an extension crate** using the
   `ExtensionOp` / `define_extension_runtime!` path, with an optional CUDA
   implementation through `cuda::raw`.
5. **`dot_general` batch dims** cover GQA and per-head attention; remember the
   batch-trailing output convention.
6. **Laya-first is feasible**: every Laya primitive maps to existing ops or
   simple compositions; only the fused kernels are extensions.
7. **Apple GPU is farther than the package assumed**: WebGPU is dot_general-only
   in practice, so the model crates must not depend on a broad WebGPU surface.

---

## 13. Open items to verify

- [ ] Exact public session-op method names for `gather`, `slice`,
      `concatenate`, and `where_select` in the direct and eager tiers.
- [ ] Whether `dot_general` on CUDA accepts the head-batched attention shapes
      directly, and the best layout for GQA.
- [ ] Performance of composed softmax/RMS versus a fused extension.
- [ ] Whether the chunked Delta `triangular_solve` runs comfortably on CUDA for
      every `value_dim` used by the checkpoint.
- [ ] A supported preallocated-output path for every hot op (to meet the
      workspace requirements).
- [ ] Reduced-precision timeline in tenferro before committing to Phase 8.
- [ ] Re-survey after any tenferro revision bump, since APIs here changed
      between the package snapshot and `471c4278` (notably session `??`).
