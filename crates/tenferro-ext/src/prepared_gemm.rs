//! Owned prepared CPU projection as a tenferro inference extension.
use cpu_kernels::prepared_projection::{PreparedProjection, ProjectionWorkspace};
use std::any::Any;
use std::hash::Hasher;
use std::sync::{Arc, Mutex};
use tenferro_ad::extension::{
    EagerExtensionTarget, apply_eager_with_targeted_extension_in_session,
};
use tenferro_ad::{EagerSession, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_runtime::extension::{
    ExtensionOp, ExtensionShapeContext, SymDim, define_extension_runtime,
};
use tenferro_runtime::{ErrorPhase, ExtensionCacheKey, ExtensionModule};
use tenferro_tensor::{BackendSession, DType, Tensor, TensorBackend, TensorRead};
const FAMILY: &str = "tenferro-decision.prepared_gemm.v1";
fn error(field: &'static str, message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::invalid_argument("tenferro-ext::prepared_gemm", field, message)
}
#[derive(Debug)]
struct State {
    projection: PreparedProjection,
}
#[derive(Debug, Default)]
struct Workspaces {
    projection: Vec<f32>,
    native: ProjectionWorkspace,
}
impl Workspaces {
    fn retained_bytes(&self) -> usize {
        self.projection
            .capacity()
            .saturating_mul(std::mem::size_of::<f32>())
            .saturating_add(self.native.retained_bytes())
    }
}
/// Packed constant inference weights, owned independently of input storage and
/// eager runtimes. Clones share exclusive CPU preparation and workspace state.
#[derive(Clone, Debug)]
pub struct PreparedGemm {
    input: usize,
    output: usize,
    state: Arc<Mutex<State>>,
}
impl PreparedGemm {
    /// Prepare row-major `(output, input)` F32 weights once.
    pub fn new(weights: &[f32], input: usize, output: usize) -> tenferro_ad::Result<Self> {
        let projection = PreparedProjection::new(weights, input, output)
            .map_err(|e| tenferro_ad::Error::TensorRuntime(error("weights", e)))?;
        Ok(Self {
            input,
            output,
            state: Arc::new(Mutex::new(State { projection })),
        })
    }
}
#[derive(Clone, Debug)]
struct PreparedGemmOp {
    weights: PreparedGemm,
    bias: bool,
    geglu: bool,
}
impl ExtensionOp for PreparedGemmOp {
    fn family_id(&self) -> &'static str {
        FAMILY
    }
    fn payload_hash(&self, h: &mut dyn Hasher) {
        h.write_usize(Arc::as_ptr(&self.weights.state) as usize);
        h.write_u8(self.bias as u8);
        h.write_u8(self.geglu as u8);
    }
    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>().is_some_and(|v| {
            Arc::ptr_eq(&self.weights.state, &v.weights.state)
                && self.bias == v.bias
                && self.geglu == v.geglu
        })
    }
    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn input_count(&self) -> usize {
        1 + usize::from(self.bias)
    }
    fn output_count(&self) -> usize {
        1
    }
    fn infer_output_meta(
        &self,
        ctx: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        let mut shape = ctx.input_shape(0)?.to_vec();
        if shape.is_empty() {
            return Err(error("shape", "expected feature-first activation"));
        }
        ctx.require_equal(shape[0].clone(), SymDim::from(self.weights.input))?;
        if self.geglu && self.weights.output % 2 != 0 {
            return Err(error("output", "GeGLU requires even projection width"));
        }
        if self.bias {
            let bias = ctx.input_shape(1)?.to_vec();
            if bias.len() != 1 {
                return Err(error("bias", "expected rank-one bias"));
            }
            ctx.require_equal(bias[0].clone(), SymDim::from(self.weights.output))?;
        }
        for i in 0..self.input_count() {
            if ctx.input_dtype(i)? != DType::F32 {
                return Err(error("dtype", "requires F32"));
            }
        }
        shape[0] = SymDim::from(if self.geglu {
            self.weights.output / 2
        } else {
            self.weights.output
        });
        Ok(vec![(DType::F32, shape)])
    }
}
fn execute(
    op: &PreparedGemmOp,
    session: &mut dyn BackendSession,
    caches: &mut tenferro_runtime::ExtensionCacheStore,
    inputs: &[TensorRead<'_>],
) -> tenferro_tensor::Result<Vec<Tensor>> {
    if inputs.len() != op.input_count() {
        return Err(error("inputs", "invalid input count"));
    }
    let mut shape = inputs[0].shape().to_vec();
    if shape.first() != Some(&op.weights.input) {
        return Err(error("shape", "invalid contracting extent"));
    }
    if inputs.iter().any(|v| v.dtype() != DType::F32) {
        return Err(error("dtype", "requires F32"));
    }
    if op.geglu && op.weights.output % 2 != 0 {
        return Err(error("output", "GeGLU requires even projection width"));
    }
    let rows = shape[1..]
        .iter()
        .try_fold(1usize, |a, b| a.checked_mul(*b))
        .ok_or_else(|| error("shape", "row extent overflow"))?;
    let extent = rows
        .checked_mul(op.weights.output)
        .ok_or_else(|| error("shape", "output extent overflow"))?;
    let storage;
    let x = match inputs[0].as_slice::<f32>() {
        Ok(v) => v,
        Err(_) => {
            storage = session.to_contiguous_read(inputs[0].clone())?;
            storage.as_slice::<f32>()?
        }
    };
    let bias_storage;
    let bias = if op.bias {
        if inputs[1].shape() != [op.weights.output] {
            return Err(error("bias", "invalid bias shape"));
        }
        Some(match inputs[1].as_slice::<f32>() {
            Ok(v) => v,
            Err(_) => {
                bias_storage = session.to_contiguous_read(inputs[1].clone())?;
                bias_storage.as_slice::<f32>()?
            }
        })
    } else {
        None
    };
    let mut state = op
        .weights
        .state
        .lock()
        .map_err(|_| error("state", "prepared projection lock poisoned"))?;
    let mut output = vec![0.; if op.geglu { extent / 2 } else { extent }];
    let key = ExtensionCacheKey::new(FAMILY, "workspaces", 0);
    if caches.get::<Workspaces>(&key).is_none() {
        caches.put_with_retained_bytes(key, Workspaces::default(), Workspaces::retained_bytes);
    }
    let workspace = caches
        .get_mut::<Workspaces>(&key)
        .ok_or_else(|| error("cache", "projection workspace unavailable"))?;
    let projected = if op.geglu {
        workspace.projection.resize(extent, 0.);
        workspace.projection.as_mut_slice()
    } else {
        output.as_mut_slice()
    };
    if let Some(b) = bias {
        for row in projected.chunks_exact_mut(op.weights.output) {
            row.copy_from_slice(b);
        }
    }
    state
        .projection
        .run_with_workspace(x, rows, projected, bias.is_some(), &mut workspace.native)
        .map_err(|e| error("execution", e))?;
    if op.geglu {
        state
            .projection
            .geglu_with_workspace(
                &workspace.projection,
                rows,
                &mut output,
                &mut workspace.native,
            )
            .map_err(|e| error("activation", e))?;
    }
    shape[0] = if op.geglu {
        op.weights.output / 2
    } else {
        op.weights.output
    };
    Ok(vec![Tensor::from_vec_col_major(shape, output)?])
}
fn supported<B: TensorBackend + 'static>(_: &PreparedGemmOp) -> bool {
    std::any::TypeId::of::<B>() == std::any::TypeId::of::<CpuBackend>()
}
define_extension_runtime! {runtime=PreparedGemmRuntime,family_id=FAMILY,op_type=PreparedGemmOp,execute_in_session=execute,session_supported=supported,backend_bound=TensorBackend,}
fn module(target: EagerExtensionTarget) -> tenferro_runtime::Result<Arc<dyn ExtensionModule>> {
    extension_module::<CpuBackend>(target.engine_id).map_err(|source| {
        tenferro_runtime::Error::runtime_state_source(
            "tenferro-ext::prepared_gemm",
            ErrorPhase::Execution,
            source,
        )
    })
}
/// Inference projection operations using owned prepared constant weights.
pub trait EagerSessionPreparedGemmExt {
    /// Feature-first projection, optional bias, and optionally exact GeGLU.
    fn prepared_gemm(
        &mut self,
        x: &EagerTensor,
        weights: &PreparedGemm,
        bias: Option<&EagerTensor>,
        geglu: bool,
    ) -> tenferro_ad::Result<EagerTensor>;
}
impl EagerSessionPreparedGemmExt for EagerSession<'_> {
    fn prepared_gemm(
        &mut self,
        x: &EagerTensor,
        weights: &PreparedGemm,
        bias: Option<&EagerTensor>,
        geglu: bool,
    ) -> tenferro_ad::Result<EagerTensor> {
        let mut inputs = vec![x];
        if let Some(b) = bias {
            inputs.push(b);
        }
        let outputs = apply_eager_with_targeted_extension_in_session(
            self,
            Arc::new(PreparedGemmOp {
                weights: weights.clone(),
                bias: bias.is_some(),
                geglu,
            }),
            &inputs,
            module,
        )?;
        if outputs.len() != 1 {
            return Err(tenferro_ad::Error::TensorRuntime(error(
                "outputs",
                "expected one output",
            )));
        }
        Ok(outputs.into_iter().next().unwrap())
    }
}
