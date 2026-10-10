//! Fused CUDA kernels for the engines' device forward (`cuda` feature).
//!
//! On the pinned tenferro CUDA provider each eager op costs a kernel launch
//! plus provider bookkeeping; the composed norms, activations and attention
//! of a small-batch forward are thousands of tiny launches, so the forward is
//! launch-bound rather than compute-bound. [`FusedScope`] replaces those
//! chains with single NVRTC-compiled kernels (`cuda/fused.cu`) launched
//! through tenferro's public raw-CUDA seam (`CudaExecSession::with_raw`) on
//! the session's own stream, so they are ordered with every native op.
//!
//! Ownership follows the raw-launch contract: a scope retains every operand
//! and output (as eager tensors) until [`FusedScope::finish`] synchronizes
//! the stream, and its `Drop` synchronizes too (or leaks the retained
//! resources if completion cannot be established). Compiled modules are
//! `!Send`, so they live in a per-thread cache keyed by runtime identity; CUDA
//! eager sessions run their callback on the calling thread. Inputs that are
//! not dense zero-offset column-major device tensors are first materialized
//! with `duplicate_value`. Only F32 is supported; there is no CPU fallback.

use std::cell::RefCell;
use std::rc::Rc;

use tenferro_ad::{DType, EagerSession, EagerTensor, Result};
use tenferro_gpu::cuda::raw::{Function, KernelArg, LaunchConfig, Module, NvrtcOptions};
use tenferro_gpu::cuda::{CudaRuntime, CudaRuntimeIdentity, with_cuda_exec_session};
use tenferro_tensor::{Tensor, TypedTensor};

/// CUDA source compiled by [`FusedScope::begin`].
pub const FUSED_KERNEL_SOURCE: &str = include_str!("cuda/fused.cu");

/// Largest sequence length the fused attention kernel accepts (its scores
/// and query live in dynamic shared memory).
pub const FUSED_ATTENTION_MAX_LENGTH: usize = 8192;

struct FusedKernels {
    runtime: CudaRuntimeIdentity,
    _module: Module,
    norm: Function,
    gated_silu: Function,
    geglu: Function,
    bias_act: Function,
    jeff_prep: Function,
    laya_prep: Function,
    attention: Function,
    embedding: Function,
    gated_silu_split: Function,
}

thread_local! {
    static KERNELS: RefCell<Vec<Rc<FusedKernels>>> = const { RefCell::new(Vec::new()) };
}

const MAX_CACHED_MODULES: usize = 4;

fn unsupported(message: impl Into<String>) -> tenferro_ad::Error {
    tenferro_ad::Error::TensorRuntime(tenferro_tensor::Error::unsupported(
        "tenferro-ext::cuda_fused",
        message.into(),
    ))
}

fn invalid(message: impl Into<String>) -> tenferro_ad::Error {
    tenferro_ad::Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
        "tenferro-ext::cuda_fused",
        "input",
        message.into(),
    ))
}

/// The NVRTC virtual architecture: `TENFERRO_CUDA_ARCH` when set, else the
/// session device's compute capability.
fn cuda_arch(session: &mut EagerSession<'_>) -> Result<String> {
    if let Ok(arch) = std::env::var("TENFERRO_CUDA_ARCH") {
        if !arch.is_empty() {
            return Ok(arch);
        }
    }
    let device =
        with_cuda_exec_session(session.backend_session(), |cuda| cuda.runtime().device_id())
            .ok_or_else(|| unsupported("CUDA execution session required"))?;
    let devices = tenferro_gpu::cuda::cuda_devices()
        .map_err(|error| unsupported(format!("device discovery: {error}")))?;
    let info = devices
        .iter()
        .find(|info| info.id() == device)
        .ok_or_else(|| unsupported("session device not discovered"))?;
    let capability = info.compute_capability();
    Ok(format!("compute_{}{}", capability.major, capability.minor))
}

fn kernels(session: &mut EagerSession<'_>) -> Result<(CudaRuntime, Rc<FusedKernels>)> {
    let (runtime, identity) = with_cuda_exec_session(session.backend_session(), |cuda| {
        (cuda.runtime().clone(), cuda.runtime().runtime_identity())
    })
    .ok_or_else(|| unsupported("CUDA execution session required"))?;
    let cached = KERNELS.with(|cache| {
        cache
            .borrow()
            .iter()
            .find(|kernels| kernels.runtime == identity)
            .cloned()
    });
    if let Some(kernels) = cached {
        return Ok((runtime, kernels));
    }
    let arch = cuda_arch(session)?;
    let kernels = with_cuda_exec_session(session.backend_session(), |cuda| {
        cuda.with_raw("fused_compile", |raw| {
            let options = NvrtcOptions {
                arch: Some(arch.clone()),
                std: Some("c++14".into()),
                // Keep multiply/add rounding identical to the CPU paths.
                extra: vec!["--fmad=false".into()],
            };
            let module = raw.compile_nvrtc(FUSED_KERNEL_SOURCE, &options)?;
            Ok(FusedKernels {
                runtime: raw.runtime_identity(),
                norm: module.function("fused_norm")?,
                gated_silu: module.function("fused_gated_silu")?,
                geglu: module.function("fused_geglu_cols")?,
                bias_act: module.function("fused_bias_act")?,
                jeff_prep: module.function("fused_jeff_qkv_prep")?,
                laya_prep: module.function("fused_laya_qkv_prep")?,
                attention: module.function("fused_attention")?,
                embedding: module.function("fused_embedding_rows")?,
                gated_silu_split: module.function("fused_gated_silu_split")?,
                _module: module,
            })
        })
    })
    .expect("CUDA session checked above")?;
    let kernels = Rc::new(kernels);
    KERNELS.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.len() >= MAX_CACHED_MODULES {
            cache.remove(0);
        }
        cache.push(kernels.clone());
    });
    Ok((runtime, kernels))
}

/// A launch input.
#[derive(Clone, Copy)]
enum In<'a> {
    /// A per-call activation (copied into a dense device tensor).
    A(&'a EagerTensor),
    /// A persistent weight (dense copy cached per thread across calls).
    W(&'a EagerTensor),
    /// A raw tensor already owned by the scope.
    R(usize),
    /// Persistent host data (e.g. model weights), uploaded once per thread
    /// and runtime and cached by storage identity; keep it alive and
    /// unchanged while the engine runs (the [`crate::TensorCache`] contract).
    H(&'a [f32], &'a [usize]),
    /// Per-call host data, uploaded for this launch.
    U(&'a [f32], &'a [usize]),
}

/// A resolved launch operand.
enum Src {
    Raw(usize),
    Eager(Box<EagerTensor>),
}

type HostKey = (CudaRuntimeIdentity, usize, usize, Vec<usize>);

thread_local! {
    static HOST_WEIGHTS: RefCell<std::collections::HashMap<HostKey, Rc<Tensor>>> =
        RefCell::new(std::collections::HashMap::new());
}

fn upload(session: &mut EagerSession<'_>, data: &[f32], shape: &[usize]) -> Result<Tensor> {
    let host = Tensor::from_vec_col_major(shape.to_vec(), data.to_vec())?;
    Ok(session
        .backend_session()
        .upload_host_tensor(tenferro_tensor::TensorRead::from_tensor(&host))?)
}

/// A dense, zero-offset, column-major device copy of `x`.
fn dense_copy(session: &mut EagerSession<'_>, x: &EagerTensor) -> Result<Tensor> {
    let copy = session
        .backend_session()
        .to_contiguous_read(x.tensor_read())?;
    let dense = copy.as_typed::<f32>().is_some_and(dense_device);
    if !dense {
        return Err(unsupported(
            "operand did not materialize as a dense device tensor",
        ));
    }
    Ok(copy)
}

struct WeightEntry {
    // Retaining the source keeps its allocation identity from being recycled.
    _source: EagerTensor,
    copy: Rc<Tensor>,
}

type WeightKey = (
    tenferro_tensor::AllocationDomainId,
    tenferro_tensor::AllocationId,
    Vec<usize>,
    Vec<isize>,
    isize,
);

thread_local! {
    static WEIGHTS: RefCell<std::collections::HashMap<WeightKey, WeightEntry>> =
        RefCell::new(std::collections::HashMap::new());
}

/// Bound on cached weight copies per thread; the cache is cleared when full.
const MAX_CACHED_WEIGHTS: usize = 4096;

/// The physical identity of an F32 device value, when it has one.
fn weight_key(x: &EagerTensor) -> Option<WeightKey> {
    let value = x.value().ok()?;
    let tenferro_tensor::TensorView::F32(view) = value.as_tensor_view() else {
        return None;
    };
    Some((
        view.allocation_domain()?,
        view.allocation_id()?,
        view.shape().to_vec(),
        view.strides().to_vec(),
        view.offset(),
    ))
}

/// A cached dense device copy of a persistent weight tensor. Weights without
/// an allocation identity are copied per call.
fn weight_copy(session: &mut EagerSession<'_>, x: &EagerTensor) -> Result<Rc<Tensor>> {
    let Some(key) = weight_key(x) else {
        return Ok(Rc::new(dense_copy(session, x)?));
    };
    if let Some(copy) = WEIGHTS.with(|cache| cache.borrow().get(&key).map(|e| e.copy.clone())) {
        return Ok(copy);
    }
    let copy = Rc::new(dense_copy(session, x)?);
    WEIGHTS.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.len() >= MAX_CACHED_WEIGHTS {
            cache.clear();
        }
        cache.insert(
            key,
            WeightEntry {
                _source: x.clone(),
                copy: copy.clone(),
            },
        );
    });
    Ok(copy)
}

/// One kernel argument, in the kernel's formal parameter order.
#[derive(Clone, Copy)]
enum Arg {
    /// Input tensor `i` (read-only device pointer).
    In(usize),
    /// Output tensor `i` (fresh device allocation).
    Out(usize),
    /// A null device pointer (optional operand absent).
    Null,
    I32(i32),
    I64(i64),
    F32(f32),
}

/// Activation applied by [`FusedScope::bias_act_first`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusedActivation {
    /// No activation (bias only).
    Identity,
    /// Exact erf-based GELU.
    GeluErf,
    /// ReLU.
    Relu,
}

/// Scoped owner of fused CUDA launches; see the module documentation.
///
/// Naturally `!Send` (it holds the thread-local module); create it inside the
/// admitted eager callback and [`finish`](Self::finish) it before returning.
pub struct FusedScope {
    runtime: CudaRuntime,
    kernels: Option<Rc<FusedKernels>>,
    retained: Vec<EagerTensor>,
    raw: Vec<Rc<Tensor>>,
    completed: bool,
}

impl Drop for FusedScope {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        if self.runtime.synchronize().is_err() {
            // Completion is unknown: keep every operand/output and the module
            // alive rather than recycling memory a queued kernel may use.
            std::mem::forget(std::mem::take(&mut self.retained));
            std::mem::forget(std::mem::take(&mut self.raw));
            std::mem::forget(self.kernels.take());
        }
    }
}

fn checked_i32(value: usize, what: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| invalid(format!("{what} exceeds the i32 kernel ABI")))
}

impl FusedScope {
    /// Start a scope on a CUDA session (compiling the module once per
    /// thread/runtime). Fails on other backends.
    pub fn begin(session: &mut EagerSession<'_>) -> Result<Self> {
        let (runtime, kernels) = kernels(session)?;
        Ok(Self {
            runtime,
            kernels: Some(kernels),
            retained: Vec::new(),
            raw: Vec::new(),
            completed: false,
        })
    }

    /// Synchronize the stream and release the retained operands.
    pub fn finish(mut self) -> Result<()> {
        self.runtime.synchronize()?;
        self.completed = true;
        Ok(())
    }

    /// The number of tensors retained until completion.
    pub fn retained(&self) -> usize {
        self.retained.len() + self.raw.len()
    }

    /// Resolve an input to an index into `self.raw`: activations are copied
    /// into a dense device tensor (eager values are pooled views, and the raw
    /// seam binds owned tensors only); weights reuse a per-thread dense copy.
    fn resolve(&mut self, session: &mut EagerSession<'_>, input: In<'_>) -> Result<Src> {
        let tensor = match input {
            In::R(index) => return Ok(Src::Raw(index)),
            In::A(x) | In::W(x) if x.dtype() != DType::F32 => {
                return Err(unsupported("fused CUDA kernels support F32 only"));
            }
            In::A(x) => {
                self.retained.push(x.clone());
                // Eager op outputs are owned dense tensors and bind in place;
                // pooled views (e.g. constants) are copied.
                let read = x.tensor_read();
                let bindable = read
                    .as_tensor()
                    .and_then(|tensor| tensor.as_typed::<f32>())
                    .is_some_and(dense_device);
                if bindable {
                    return Ok(Src::Eager(Box::new(x.clone())));
                }
                Rc::new(dense_copy(session, x)?)
            }
            In::W(x) => weight_copy(session, x)?,
            In::H(data, shape) => {
                if data.len() != shape.iter().product::<usize>() {
                    return Err(invalid("host data does not match its shape"));
                }
                let key = (
                    self.runtime.runtime_identity(),
                    data.as_ptr() as usize,
                    data.len(),
                    shape.to_vec(),
                );
                match HOST_WEIGHTS.with(|cache| cache.borrow().get(&key).cloned()) {
                    Some(tensor) => tensor,
                    None => {
                        let tensor = Rc::new(upload(session, data, shape)?);
                        HOST_WEIGHTS.with(|cache| {
                            let mut cache = cache.borrow_mut();
                            if cache.len() >= MAX_CACHED_WEIGHTS {
                                cache.clear();
                            }
                            cache.insert(key, tensor.clone());
                        });
                        tensor
                    }
                }
            }
            In::U(data, shape) => {
                if data.len() != shape.iter().product::<usize>() {
                    return Err(invalid("host data does not match its shape"));
                }
                Rc::new(upload(session, data, shape)?)
            }
        };
        self.raw.push(tensor);
        Ok(Src::Raw(self.raw.len() - 1))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_core(
        &mut self,
        session: &mut EagerSession<'_>,
        op: &'static str,
        function: fn(&FusedKernels) -> &Function,
        inputs: &[In<'_>],
        outputs: &[Vec<usize>],
        config: LaunchConfig,
        args: &[Arg],
    ) -> Result<Vec<TypedTensor<f32>>> {
        let sources = inputs
            .iter()
            .map(|input| self.resolve(session, *input))
            .collect::<Result<Vec<_>>>()?;
        let kernels = self.kernels.clone().expect("live module");
        let reads: Vec<_> = sources
            .iter()
            .map(|source| match source {
                Src::Eager(x) => Some(x.tensor_read()),
                Src::Raw(_) => None,
            })
            .collect();
        let typed: Vec<&TypedTensor<f32>> = sources
            .iter()
            .zip(&reads)
            .map(|(source, read)| {
                let tensor = match (source, read) {
                    (Src::Raw(index), _) => Some(&*self.raw[*index]),
                    (Src::Eager(_), Some(read)) => read.as_tensor(),
                    (Src::Eager(_), None) => None,
                };
                tensor
                    .and_then(|tensor| tensor.as_typed::<f32>())
                    .ok_or_else(|| unsupported("fused operand must be a dense F32 tensor"))
            })
            .collect::<Result<_>>()?;
        with_cuda_exec_session(session.backend_session(), |cuda| {
            cuda.with_raw(op, |raw| {
                let mut outs = outputs
                    .iter()
                    .map(|shape| raw.alloc_output::<f32>(shape))
                    .collect::<tenferro_tensor::Result<Vec<_>>>()?;
                {
                    let in_refs = typed
                        .iter()
                        .map(|tensor| raw.tensor(*tensor))
                        .collect::<tenferro_tensor::Result<Vec<_>>>()?;
                    let out_refs = outs
                        .iter_mut()
                        .map(|tensor| raw.tensor_mut(tensor))
                        .collect::<tenferro_tensor::Result<Vec<_>>>()?;
                    let kernel_args: Vec<KernelArg<'_>> = args
                        .iter()
                        .map(|arg| match *arg {
                            Arg::In(i) => KernelArg::tensor(&in_refs[i]),
                            Arg::Out(i) => KernelArg::tensor_mut(&out_refs[i]),
                            Arg::Null => KernelArg::u64(0),
                            Arg::I32(v) => KernelArg::i32(v),
                            Arg::I64(v) => KernelArg::i64(v),
                            Arg::F32(v) => KernelArg::f32(v),
                        })
                        .collect();
                    // SAFETY: the argument list follows the kernel's formal
                    // parameters (fused.cu); callers validate the shapes the
                    // launch geometry indexes; outputs are fresh allocations
                    // disjoint from every input; and this scope retains all
                    // operands, outputs and the module until its completion
                    // fence (or leaks them when completion is unknown).
                    unsafe { raw.launch(function(&kernels), config, &kernel_args) }?;
                }
                Ok(outs)
            })
        })
        .ok_or_else(|| unsupported("CUDA execution session required"))?
        .map_err(Into::into)
    }

    /// Launch and register the outputs as eager tensors.
    #[allow(clippy::too_many_arguments)]
    fn launch(
        &mut self,
        session: &mut EagerSession<'_>,
        op: &'static str,
        function: fn(&FusedKernels) -> &Function,
        inputs: &[In<'_>],
        outputs: &[Vec<usize>],
        config: LaunchConfig,
        args: &[Arg],
    ) -> Result<Vec<EagerTensor>> {
        let outputs = self.launch_core(session, op, function, inputs, outputs, config, args)?;
        let mut eager = Vec::with_capacity(outputs.len());
        for output in outputs {
            let tensor = session.constant_from(Tensor::from_typed(output))?;
            self.retained.push(tensor.clone());
            eager.push(tensor);
        }
        Ok(eager)
    }

    /// Launch and keep the outputs as scope-owned raw tensors (for a fused
    /// consumer), returning their [`In::R`] indices.
    #[allow(clippy::too_many_arguments)]
    fn launch_raw(
        &mut self,
        session: &mut EagerSession<'_>,
        op: &'static str,
        function: fn(&FusedKernels) -> &Function,
        inputs: &[In<'_>],
        outputs: &[Vec<usize>],
        config: LaunchConfig,
        args: &[Arg],
    ) -> Result<Vec<usize>> {
        let outputs = self.launch_core(session, op, function, inputs, outputs, config, args)?;
        Ok(outputs
            .into_iter()
            .map(|output| {
                self.raw.push(Rc::new(Tensor::from_typed(output)));
                self.raw.len() - 1
            })
            .collect())
    }

    #[allow(clippy::too_many_arguments)]
    fn norm(
        &mut self,
        session: &mut EagerSession<'_>,
        x: &EagerTensor,
        weight: &EagerTensor,
        bias: Option<&EagerTensor>,
        layout: (usize, usize, usize, usize),
        eps: f32,
        mode: i32,
    ) -> Result<EagerTensor> {
        let (vectors, width, vec_stride, feat_stride) = layout;
        if weight.shape() != [width] || bias.is_some_and(|bias| bias.shape() != [width]) {
            return Err(invalid("norm weight/bias must have shape [width]"));
        }
        let threads = if width <= 128 { 128 } else { 256 };
        let mut inputs = vec![In::A(x), In::W(weight)];
        let bias_arg = match bias {
            Some(bias) => {
                inputs.push(In::W(bias));
                Arg::In(2)
            }
            None => Arg::Null,
        };
        let out = self.launch(
            session,
            "fused_norm",
            |k| &k.norm,
            &inputs,
            &[x.shape().to_vec()],
            LaunchConfig {
                grid: [checked_i32(vectors, "vectors")? as u32, 1, 1],
                block: [threads, 1, 1],
                shared_mem_bytes: 0,
            },
            &[
                Arg::Out(0),
                Arg::In(0),
                Arg::In(1),
                bias_arg,
                Arg::I32(checked_i32(vectors, "vectors")?),
                Arg::I32(checked_i32(width, "width")?),
                Arg::I32(checked_i32(vec_stride, "stride")?),
                Arg::I32(checked_i32(feat_stride, "stride")?),
                Arg::F32(eps),
                Arg::I32(mode),
            ],
        )?;
        Ok(out.into_iter().next().expect("one output"))
    }

    /// RMSNorm over the **last** axis of `x` (any rank ≥ 1): `x · w`, or
    /// `x · (1 + w)` when `centered` (Qwen3.5).
    pub fn rms_norm_last(
        &mut self,
        session: &mut EagerSession<'_>,
        x: &EagerTensor,
        weight: &EagerTensor,
        centered: bool,
        eps: f64,
    ) -> Result<EagerTensor> {
        let (&width, leading) = x
            .shape()
            .split_last()
            .ok_or_else(|| invalid("x must have a feature axis"))?;
        let vectors: usize = leading.iter().product();
        if vectors == 0 || width == 0 {
            return Err(invalid("empty input"));
        }
        self.norm(
            session,
            x,
            weight,
            None,
            (vectors, width, 1, vectors),
            eps as f32,
            if centered { 1 } else { 0 },
        )
    }

    /// LayerNorm over the **first** axis of a feature-first `x (d, ...)`.
    pub fn layer_norm_first(
        &mut self,
        session: &mut EagerSession<'_>,
        x: &EagerTensor,
        weight: &EagerTensor,
        bias: Option<&EagerTensor>,
        eps: f64,
    ) -> Result<EagerTensor> {
        let (&width, trailing) = x
            .shape()
            .split_first()
            .ok_or_else(|| invalid("x must have a feature axis"))?;
        let vectors: usize = trailing.iter().product();
        if vectors == 0 || width == 0 {
            return Err(invalid("empty input"));
        }
        self.norm(
            session,
            x,
            weight,
            bias,
            (vectors, width, width, 1),
            eps as f32,
            2,
        )
    }

    fn elementwise_config(n: usize) -> Result<LaunchConfig> {
        let blocks = n.div_ceil(256);
        Ok(LaunchConfig {
            grid: [checked_i32(blocks, "elements")? as u32, 1, 1],
            block: [256, 1, 1],
            shared_mem_bytes: 0,
        })
    }

    /// `silu(gate) · up`, elementwise (same shapes).
    pub fn gated_silu(
        &mut self,
        session: &mut EagerSession<'_>,
        gate: &EagerTensor,
        up: &EagerTensor,
    ) -> Result<EagerTensor> {
        if gate.shape() != up.shape() {
            return Err(invalid("gate and up must have the same shape"));
        }
        let n: usize = gate.shape().iter().product();
        let out = self.launch(
            session,
            "fused_gated_silu",
            |k| &k.gated_silu,
            &[In::A(gate), In::A(up)],
            &[gate.shape().to_vec()],
            Self::elementwise_config(n)?,
            &[
                Arg::Out(0),
                Arg::In(0),
                Arg::In(1),
                Arg::I32(checked_i32(n, "elements")?),
            ],
        )?;
        Ok(out.into_iter().next().expect("one output"))
    }

    /// GeGLU over a feature-first `u (2·inter, ...)`: `gelu_erf(u[:inter]) ·
    /// u[inter:]`, returning `(inter, ...)`.
    pub fn geglu_first(
        &mut self,
        session: &mut EagerSession<'_>,
        u: &EagerTensor,
        inter: usize,
    ) -> Result<EagerTensor> {
        let (&rows, trailing) = u
            .shape()
            .split_first()
            .ok_or_else(|| invalid("u must have a feature axis"))?;
        if rows != 2 * inter || inter == 0 {
            return Err(invalid("u must have 2 * inter rows"));
        }
        let cols: usize = trailing.iter().product();
        let mut shape = u.shape().to_vec();
        shape[0] = inter;
        let out = self.launch(
            session,
            "fused_geglu",
            |k| &k.geglu,
            &[In::A(u)],
            &[shape],
            Self::elementwise_config(inter * cols)?,
            &[
                Arg::Out(0),
                Arg::In(0),
                Arg::I32(checked_i32(inter, "inter")?),
                Arg::I32(checked_i32(cols, "cols")?),
            ],
        )?;
        Ok(out.into_iter().next().expect("one output"))
    }

    /// `act(x + bias)` over a feature-first `x (rows, ...)` with an optional
    /// `bias (rows,)`.
    pub fn bias_act_first(
        &mut self,
        session: &mut EagerSession<'_>,
        x: &EagerTensor,
        bias: Option<&EagerTensor>,
        act: FusedActivation,
    ) -> Result<EagerTensor> {
        let (&rows, trailing) = x
            .shape()
            .split_first()
            .ok_or_else(|| invalid("x must have a feature axis"))?;
        if bias.is_some_and(|bias| bias.shape() != [rows]) {
            return Err(invalid("bias must have shape [rows]"));
        }
        let cols: usize = trailing.iter().product();
        let mut inputs = vec![In::A(x)];
        let bias_arg = match bias {
            Some(bias) => {
                inputs.push(In::W(bias));
                Arg::In(1)
            }
            None => Arg::Null,
        };
        let out = self.launch(
            session,
            "fused_bias_act",
            |k| &k.bias_act,
            &inputs,
            &[x.shape().to_vec()],
            Self::elementwise_config(rows * cols)?,
            &[
                Arg::Out(0),
                Arg::In(0),
                bias_arg,
                Arg::I32(checked_i32(rows, "rows")?),
                Arg::I32(checked_i32(cols, "cols")?),
                Arg::I32(match act {
                    FusedActivation::Identity => 0,
                    FusedActivation::GeluErf => 1,
                    FusedActivation::Relu => 2,
                }),
            ],
        )?;
        Ok(out.into_iter().next().expect("one output"))
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &mut self,
        session: &mut EagerSession<'_>,
        heads_qkv: &[usize],
        bias: Option<&EagerTensor>,
        active: Option<&EagerTensor>,
        gate: Option<&EagerTensor>,
        dims: (usize, usize, usize, usize),
        causal: bool,
        out_shape: Vec<usize>,
        strides: (usize, usize, usize),
        gate_offset: usize,
    ) -> Result<EagerTensor> {
        let (length, heads, head_dim, batch) = dims;
        if length > FUSED_ATTENTION_MAX_LENGTH {
            return Err(unsupported(
                "sequence too long for the fused attention kernel",
            ));
        }
        let mut inputs: Vec<In<'_>> = heads_qkv.iter().map(|&index| In::R(index)).collect();
        let mut optional_args = [Arg::Null; 3];
        for (slot, tensor) in optional_args.iter_mut().zip([bias, active, gate]) {
            if let Some(tensor) = tensor {
                inputs.push(In::A(tensor));
                *slot = Arg::In(inputs.len() - 1);
            }
        }
        let [bias_arg, active_arg, gate_arg] = optional_args;
        let threads = 128u32;
        let out = self.launch(
            session,
            "fused_attention",
            |k| &k.attention,
            &inputs,
            &[out_shape],
            LaunchConfig {
                grid: [
                    checked_i32(length, "length")? as u32,
                    checked_i32(heads, "heads")? as u32,
                    checked_i32(batch, "batch")? as u32,
                ],
                block: [threads, 1, 1],
                shared_mem_bytes: ((length + head_dim) * size_of::<f32>()) as u32,
            },
            &[
                Arg::Out(0),
                Arg::In(0),
                Arg::In(1),
                Arg::In(2),
                bias_arg,
                active_arg,
                gate_arg,
                Arg::I32(checked_i32(length, "length")?),
                Arg::I32(checked_i32(heads, "heads")?),
                Arg::I32(checked_i32(head_dim, "head_dim")?),
                Arg::I32(checked_i32(batch, "batch")?),
                Arg::F32(1.0 / (head_dim as f32).sqrt()),
                Arg::I32(causal as i32),
                Arg::I64(strides.0 as i64),
                Arg::I64(strides.1 as i64),
                Arg::I64(strides.2 as i64),
                Arg::I64(gate_offset as i64),
            ],
        )?;
        Ok(out.into_iter().next().expect("one output"))
    }

    /// Gather token rows from a host **row-major `(hidden, vocab)`** table
    /// (uploaded once and cached by storage identity, in its original layout)
    /// into a time-first `(length, hidden)`. Ids must be below `vocab` < 2^24.
    pub fn embedding_rows(
        &mut self,
        session: &mut EagerSession<'_>,
        table: &[f32],
        hidden: usize,
        vocab: usize,
        ids: &[i64],
    ) -> Result<EagerTensor> {
        let length = ids.len();
        self.embedding(
            session,
            table,
            (hidden, vocab),
            ids,
            vec![length, hidden],
            (1, length, vocab, 1),
        )
    }

    /// Gather token columns from a host **column-major `(hidden, vocab)`**
    /// table (each token's vector contiguous) into a feature-first
    /// `(hidden, ids.len())`.
    pub fn embedding_columns(
        &mut self,
        session: &mut EagerSession<'_>,
        table: &[f32],
        hidden: usize,
        vocab: usize,
        ids: &[i64],
    ) -> Result<EagerTensor> {
        let length = ids.len();
        self.embedding(
            session,
            table,
            (hidden, vocab),
            ids,
            vec![hidden, length],
            (hidden, 1, 1, hidden),
        )
    }

    fn embedding(
        &mut self,
        session: &mut EagerSession<'_>,
        table: &[f32],
        (hidden, vocab): (usize, usize),
        ids: &[i64],
        out_shape: Vec<usize>,
        (o_tok, o_feat, t_feat, t_tok): (usize, usize, usize, usize),
    ) -> Result<EagerTensor> {
        let length = ids.len();
        if table.len() != hidden * vocab || length == 0 || vocab > 1 << 24 {
            return Err(invalid("embedding table/ids shape"));
        }
        if ids.iter().any(|&id| id < 0 || id as usize >= vocab) {
            return Err(invalid("token id is outside the vocabulary"));
        }
        let ids: Vec<f32> = ids.iter().map(|&id| id as f32).collect();
        let table_shape = [hidden * vocab];
        let ids_shape = [length];
        let out = self.launch(
            session,
            "fused_embedding",
            |k| &k.embedding,
            &[In::H(table, &table_shape), In::U(&ids, &ids_shape)],
            &[out_shape],
            Self::elementwise_config(length * hidden)?,
            &[
                Arg::Out(0),
                Arg::In(0),
                Arg::In(1),
                Arg::I32(checked_i32(length, "length")?),
                Arg::I32(checked_i32(hidden, "hidden")?),
                Arg::I64(o_tok as i64),
                Arg::I64(o_feat as i64),
                Arg::I64(t_feat as i64),
                Arg::I64(t_tok as i64),
            ],
        )?;
        Ok(out.into_iter().next().expect("one output"))
    }

    /// Jeff (Qwen3.5) full attention core on the stacked time-first projection
    /// `qkvg (length, 4·heads·head_dim)` (column blocks q | k | v | gate):
    /// centered per-head RMSNorm of q/k with `q_norm`/`k_norm (head_dim,)`,
    /// partial RoPE over the first `rotary_dim` channels, causal softmax
    /// attention over keys whose `active (length,)` entry is non-zero, and the
    /// sigmoid gate. Returns the merged `(length, heads·head_dim)` before the
    /// output projection.
    #[allow(clippy::too_many_arguments)]
    pub fn jeff_attention(
        &mut self,
        session: &mut EagerSession<'_>,
        qkvg: &EagerTensor,
        q_norm: &EagerTensor,
        k_norm: &EagerTensor,
        active: &EagerTensor,
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        theta: f64,
        eps: f64,
    ) -> Result<EagerTensor> {
        let [length, stacked] = qkvg.shape() else {
            return Err(invalid("qkvg must be (length, 4 * width)"));
        };
        let (length, width) = (*length, heads * head_dim);
        if *stacked != 4 * width
            || q_norm.shape() != [head_dim]
            || k_norm.shape() != [head_dim]
            || active.shape() != [length]
            || rotary_dim > head_dim
            || rotary_dim % 2 != 0
            || head_dim > 1024
        {
            return Err(invalid("inconsistent Jeff attention shapes"));
        }
        let threads = head_dim.div_ceil(32) * 32;
        let head_major = vec![heads, length, head_dim];
        let (cos, sin) = rope_tables(length, rotary_dim / 2, |i| {
            theta.powf(-2.0 * i as f64 / rotary_dim as f64)
        });
        let table_shape = [cos.len()];
        let mut prepared = self.launch_raw(
            session,
            "fused_jeff_qkv_prep",
            |k| &k.jeff_prep,
            &[
                In::A(qkvg),
                In::W(q_norm),
                In::W(k_norm),
                In::U(&cos, &table_shape),
                In::U(&sin, &table_shape),
            ],
            &[head_major.clone(), head_major.clone(), head_major],
            LaunchConfig {
                grid: [checked_i32(length, "length")? as u32, heads as u32, 3],
                block: [threads as u32, 1, 1],
                shared_mem_bytes: (threads * size_of::<f32>()) as u32,
            },
            &[
                Arg::Out(0),
                Arg::Out(1),
                Arg::Out(2),
                Arg::In(0),
                Arg::In(1),
                Arg::In(2),
                Arg::In(3),
                Arg::In(4),
                Arg::I32(checked_i32(length, "length")?),
                Arg::I32(checked_i32(heads, "heads")?),
                Arg::I32(checked_i32(head_dim, "head_dim")?),
                Arg::I32(checked_i32(rotary_dim, "rotary_dim")?),
                Arg::F32(eps as f32),
            ],
        )?;
        // The gate is read in place from the stacked projection.
        prepared.truncate(3);
        self.attention(
            session,
            &prepared,
            None,
            Some(active),
            Some(qkvg),
            (length, heads, head_dim, 1),
            true,
            vec![length, width],
            (1, length, 0),
            3 * width * length,
        )
    }

    /// `silu(gate) · up` for a stacked `gu (rows, 2·inter)` with column
    /// blocks gate | up, returning `(rows, inter)`.
    pub fn gated_silu_stacked(
        &mut self,
        session: &mut EagerSession<'_>,
        gu: &EagerTensor,
    ) -> Result<EagerTensor> {
        let [rows, stacked] = gu.shape() else {
            return Err(invalid("gu must be (rows, 2 * inter)"));
        };
        let (rows, inter) = (*rows, *stacked / 2);
        if *stacked != 2 * inter || inter == 0 {
            return Err(invalid("gu must have an even, non-zero column count"));
        }
        let out = self.launch(
            session,
            "fused_gated_silu_split",
            |k| &k.gated_silu_split,
            &[In::A(gu)],
            &[vec![rows, inter]],
            Self::elementwise_config(rows * inter)?,
            &[
                Arg::Out(0),
                Arg::In(0),
                Arg::I32(checked_i32(rows, "rows")?),
                Arg::I32(checked_i32(inter, "inter")?),
            ],
        )?;
        Ok(out.into_iter().next().expect("one output"))
    }

    /// Laya (ModernBERT) attention core on a feature-first
    /// `qkv (3·hidden, length, batch)`: split, full-width rotate-half RoPE
    /// when `rope_base > 0`, and softmax attention with an additive
    /// `bias (length, length, batch)`. Returns `(hidden, length, batch)`
    /// before the output projection.
    pub fn laya_attention(
        &mut self,
        session: &mut EagerSession<'_>,
        qkv: &EagerTensor,
        bias: &EagerTensor,
        hidden: usize,
        heads: usize,
        rope_base: f64,
    ) -> Result<EagerTensor> {
        let [rows, length, batch] = qkv.shape() else {
            return Err(invalid("qkv must be (3 * hidden, length, batch)"));
        };
        let (length, batch) = (*length, *batch);
        let head_dim = hidden / heads;
        if *rows != 3 * hidden
            || hidden % heads != 0
            || head_dim % 2 != 0
            || head_dim > 1024
            || bias.shape() != [length, length, batch]
        {
            return Err(invalid("inconsistent Laya attention shapes"));
        }
        let threads = head_dim.div_ceil(32) * 32;
        let head_major = vec![batch, heads, length, head_dim];
        let half = head_dim / 2;
        let (cos, sin) = if rope_base > 0.0 {
            rope_tables(length, half, |i| rope_base.powf(-(i as f64) / half as f64))
        } else {
            (vec![0.0], vec![0.0])
        };
        let table_shape = [cos.len()];
        let prepared = self.launch_raw(
            session,
            "fused_laya_qkv_prep",
            |k| &k.laya_prep,
            &[
                In::A(qkv),
                In::U(&cos, &table_shape),
                In::U(&sin, &table_shape),
            ],
            &[head_major.clone(), head_major.clone(), head_major],
            LaunchConfig {
                grid: [
                    checked_i32(length, "length")? as u32,
                    heads as u32,
                    checked_i32(3 * batch, "batch")? as u32,
                ],
                block: [threads as u32, 1, 1],
                shared_mem_bytes: (threads * size_of::<f32>()) as u32,
            },
            &[
                Arg::Out(0),
                Arg::Out(1),
                Arg::Out(2),
                Arg::In(0),
                Arg::In(1),
                Arg::In(2),
                Arg::I32(checked_i32(hidden, "hidden")?),
                Arg::I32(checked_i32(heads, "heads")?),
                Arg::I32(checked_i32(length, "length")?),
                Arg::I32(checked_i32(batch, "batch")?),
                Arg::I32((rope_base > 0.0) as i32),
            ],
        )?;
        self.attention(
            session,
            &prepared,
            Some(bias),
            None,
            None,
            (length, heads, head_dim, batch),
            false,
            vec![hidden, length, batch],
            (hidden, 1, hidden * length),
            0,
        )
    }
}

/// RoPE `(length, half)` row-major cos/sin tables: angle `t · inv_freq(i)`
/// computed in f64 and rounded to f32, as the native composition does.
fn rope_tables(
    length: usize,
    half: usize,
    inv_freq: impl Fn(usize) -> f64,
) -> (Vec<f32>, Vec<f32>) {
    let freqs: Vec<f64> = (0..half).map(inv_freq).collect();
    let mut cos = Vec::with_capacity(length * half);
    let mut sin = Vec::with_capacity(length * half);
    for t in 0..length {
        for freq in &freqs {
            let angle = t as f64 * freq;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    if cos.is_empty() {
        // Keep a valid (unused) allocation for an empty rotary span.
        return (vec![0.0], vec![0.0]);
    }
    (cos, sin)
}

fn dense_device(tensor: &TypedTensor<f32>) -> bool {
    tensor.backend_buffer().is_some()
        && tensor.is_col_major_contiguous().unwrap_or(false)
        && tensor
            .layout_linear_offset(&vec![0; tensor.shape().len()])
            .is_ok_and(|offset| offset == 0)
}
