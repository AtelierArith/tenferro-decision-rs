//! Fused CPU dense projection, optional bias and exact GeGLU.
//! Retains the existing library GEMM and activation arithmetic while avoiding
//! an eager intermediate with twice the output feature width.

use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

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

/// Stable family identifier for the `gemm_geglu` extension op.
pub const GEMM_GEGLU_FAMILY_ID: &str = "tenferro-decision.gemm_geglu.v1";

/// Dense projection and GeGLU over the leading feature axis, with optional bias.
#[derive(Clone, Debug)]
pub struct GemmGegluOp {
    /// Half the projected feature extent.
    pub intermediate: usize,
    /// Whether the third input is a projected-feature bias.
    pub has_bias: bool,
}

impl GemmGegluOp {
    fn width(&self) -> tenferro_tensor::Result<usize> {
        self.intermediate
            .checked_mul(2)
            .ok_or_else(|| tensor_error("intermediate", "projected feature extent overflow"))
    }
}

fn tensor_error(field: &'static str, message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::invalid_argument("tenferro-ext::gemm_geglu", field, message)
}

impl ExtensionOp for GemmGegluOp {
    fn family_id(&self) -> &'static str {
        GEMM_GEGLU_FAMILY_ID
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_usize(self.intermediate);
        hasher.write_u8(u8::from(self.has_bias));
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<GemmGegluOp>()
            .is_some_and(|other| {
                other.intermediate == self.intermediate && other.has_bias == self.has_bias
            })
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        2 + usize::from(self.has_bias)
    }

    fn output_count(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        ctx: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        let x_shape = ctx.input_shape(0)?;
        let weight_shape = ctx.input_shape(1)?;
        if x_shape.is_empty() || weight_shape.len() != 2 {
            return Err(tensor_error(
                "shape",
                "gemm_geglu expects a feature-first activation and rank-two weight",
            ));
        }
        let mut output_shape = x_shape.to_vec();
        let x_in = x_shape[0].clone();
        let (weight_in, out) = (weight_shape[0].clone(), weight_shape[1].clone());
        ctx.require_equal(x_in, weight_in)?;
        ctx.require_equal(out.clone(), SymDim::from(self.width()?))?;
        if self.has_bias {
            let bias_shape = ctx.input_shape(2)?;
            if bias_shape.len() != 1 {
                return Err(tensor_error("bias", "expected rank-one bias"));
            }
            let bias_extent = bias_shape[0].clone();
            ctx.require_equal(out, bias_extent)?;
        }
        for index in 0..self.input_count() {
            if ctx.input_dtype(index)? != DType::F32 {
                return Err(tensor_error("dtype", "gemm_geglu requires F32 operands"));
            }
        }
        output_shape[0] = SymDim::from(self.intermediate);
        Ok(vec![(DType::F32, output_shape)])
    }
}

fn execute_gemm_geglu_in_session(
    op: &GemmGegluOp,
    session: &mut dyn BackendSession,
    caches: &mut tenferro_runtime::ExtensionCacheStore,
    inputs: &[TensorRead<'_>],
) -> tenferro_tensor::Result<Vec<Tensor>> {
    if inputs.len() != op.input_count() {
        return Err(tensor_error(
            "inputs",
            format!(
                "expected {} inputs, found {}",
                op.input_count(),
                inputs.len()
            ),
        ));
    }
    let x_shape = inputs[0].shape().to_vec();
    let weight_shape = inputs[1].shape().to_vec();
    if x_shape.is_empty() || weight_shape.len() != 2 {
        return Err(tensor_error(
            "shape",
            "gemm_geglu expects a feature-first activation and rank-two weight",
        ));
    }
    let in_dim = x_shape[0];
    let length = x_shape[1..]
        .iter()
        .try_fold(1usize, |n, value| n.checked_mul(*value))
        .ok_or_else(|| tensor_error("shape", "trailing extent overflow"))?;
    let (weight_in, out_dim) = (weight_shape[0], weight_shape[1]);
    if in_dim != weight_in {
        return Err(tensor_error(
            "shape",
            format!("x has {in_dim} contracting rows but weight has {weight_in}"),
        ));
    }
    if out_dim != op.width()? {
        return Err(tensor_error(
            "weight",
            "projected extent must be twice intermediate",
        ));
    }
    if op.has_bias && inputs[2].shape() != [out_dim] {
        return Err(tensor_error("bias", "expected projected-feature vector"));
    }
    if inputs.iter().any(|input| input.dtype() != DType::F32) {
        return Err(tensor_error("dtype", "gemm_geglu requires F32 operands"));
    }
    let x_storage;
    let weight_storage;
    let bias_storage;
    let x_data = match inputs[0].as_slice::<f32>() {
        Ok(data) => data,
        Err(_) => {
            x_storage = session.to_contiguous_read(inputs[0].clone())?;
            x_storage
                .as_slice::<f32>()
                .map_err(|error| tensor_error("x", error.to_string()))?
        }
    };
    let weight_data = match inputs[1].as_slice::<f32>() {
        Ok(data) => data,
        Err(_) => {
            weight_storage = session.to_contiguous_read(inputs[1].clone())?;
            weight_storage
                .as_slice::<f32>()
                .map_err(|error| tensor_error("weight", error.to_string()))?
        }
    };
    let bias_data = if op.has_bias {
        Some(match inputs[2].as_slice::<f32>() {
            Ok(data) => data,
            Err(_) => {
                bias_storage = session.to_contiguous_read(inputs[2].clone())?;
                bias_storage
                    .as_slice::<f32>()
                    .map_err(|error| tensor_error("bias", error.to_string()))?
            }
        })
    } else {
        None
    };
    let extent = out_dim
        .checked_mul(length)
        .ok_or_else(|| tensor_error("shape", "output extent overflow"))?;
    if let Some(bias) = bias_data {
        if bias.len() != out_dim {
            return Err(tensor_error("bias", "bias extent mismatch"));
        }
    }
    // The expanded projection is scratch, never a retained eager output.
    // Keep one buffer per runtime/family and report its actual capacity.
    let key = ExtensionCacheKey::new(GEMM_GEGLU_FAMILY_ID, "projection_scratch", 0);
    if caches.get::<Vec<f32>>(&key).is_none() {
        caches.put_with_retained_bytes(key, Vec::<f32>::new(), |values| {
            values.capacity().saturating_mul(std::mem::size_of::<f32>())
        });
    }
    let y = caches
        .get_mut::<Vec<f32>>(&key)
        .expect("projection scratch just inserted");
    y.resize(extent, 0.0);
    if let Some(bias_data) = bias_data {
        cpu_kernels::input_mul_weight_transpose_bias_into(
            x_data,
            length,
            in_dim,
            weight_data,
            out_dim,
            bias_data,
            y,
        );
    } else {
        cpu_kernels::input_mul_weight_transpose_into(
            x_data,
            length,
            in_dim,
            weight_data,
            out_dim,
            y,
        );
    }
    let mut activated = vec![0.0f32; extent / 2];
    cpu_kernels::geglu_into(y, op.intermediate, length, &mut activated);
    let mut output_shape = x_shape;
    output_shape[0] = op.intermediate;
    Ok(vec![Tensor::from_vec_col_major(output_shape, activated)?])
}

fn gemm_geglu_session_supported<B: TensorBackend + 'static>(_op: &GemmGegluOp) -> bool {
    std::any::TypeId::of::<B>() == std::any::TypeId::of::<CpuBackend>()
}

define_extension_runtime! {
    runtime = GemmGegluRuntime,
    family_id = GEMM_GEGLU_FAMILY_ID,
    op_type = GemmGegluOp,
    execute_in_session = execute_gemm_geglu_in_session,
    session_supported = gemm_geglu_session_supported,
    backend_bound = TensorBackend,
}

pub(crate) fn gemm_geglu_extension_module(
    target: EagerExtensionTarget,
) -> tenferro_runtime::Result<Arc<dyn ExtensionModule>> {
    extension_module::<CpuBackend>(target.engine_id).map_err(|source| {
        tenferro_runtime::Error::runtime_state_source(
            "tenferro-ext::gemm_geglu",
            ErrorPhase::Execution,
            source,
        )
    })
}

/// Eager-session method for the `gemm_geglu` extension.
pub trait EagerSessionGemmGegluExt {
    /// Project F32 `x (in, ...)` to `2 * intermediate` features, add optional
    /// bias, then return `gelu(first half) * second half` with the same
    /// trailing axes. Unsupported dtype/shape/backend returns an error.
    fn gemm_geglu(
        &mut self,
        x: &EagerTensor,
        weight: &EagerTensor,
        bias: Option<&EagerTensor>,
        intermediate: usize,
    ) -> tenferro_ad::Result<EagerTensor>;
}

impl EagerSessionGemmGegluExt for EagerSession<'_> {
    fn gemm_geglu(
        &mut self,
        x: &EagerTensor,
        weight: &EagerTensor,
        bias: Option<&EagerTensor>,
        intermediate: usize,
    ) -> tenferro_ad::Result<EagerTensor> {
        let mut inputs = vec![x, weight];
        if let Some(bias) = bias {
            inputs.push(bias);
        }
        let mut outputs = apply_eager_with_targeted_extension_in_session(
            self,
            Arc::new(GemmGegluOp {
                intermediate,
                has_bias: bias.is_some(),
            }),
            &inputs,
            gemm_geglu_extension_module,
        )?
        .into_iter();
        match (outputs.next(), outputs.next()) {
            (Some(value), None) => Ok(value),
            _ => Err(tenferro_ad::Error::TensorRuntime(tensor_error(
                "outputs",
                "extension returned an unexpected number of outputs",
            ))),
        }
    }
}
