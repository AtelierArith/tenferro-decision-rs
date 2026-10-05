//! Self-hosted eager CPU GEMM-with-bias extension op backed by `cpu-kernels`.
//!
//! `y = weightᵀ · x + bias` with `x` `(in, length)`, `weight` `(in, out)`, and a
//! `(out,)` bias broadcast over the `length` axis. Folding the bias into the
//! GEMM avoids a `reshape + broadcast_in_dim + add` over the full activation.

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
use tenferro_runtime::{ErrorPhase, ExtensionModule};
use tenferro_tensor::{BackendSession, DType, Tensor, TensorBackend, TensorRead};

/// Stable family identifier for the `gemm_bias` extension op.
pub const GEMM_BIAS_FAMILY_ID: &str = "tenferro-decision.gemm_bias.v1";

/// The dense `y = weightᵀ · x + bias` extension op (three inputs, one output).
#[derive(Clone, Debug)]
pub struct GemmBiasOp;

fn tensor_error(field: &'static str, message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::invalid_argument("tenferro-ext::gemm_bias", field, message)
}

impl ExtensionOp for GemmBiasOp {
    fn family_id(&self) -> &'static str {
        GEMM_BIAS_FAMILY_ID
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<GemmBiasOp>().is_some()
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        3
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
        if x_shape.len() != 2 || weight_shape.len() != 2 {
            return Err(tensor_error("shape", "gemm_bias expects two rank-2 inputs"));
        }
        let (x_in, length) = (x_shape[0].clone(), x_shape[1].clone());
        let (weight_in, out) = (weight_shape[0].clone(), weight_shape[1].clone());
        ctx.require_equal(x_in, weight_in)?;
        Ok(vec![(ctx.input_dtype(0)?, vec![out, length])])
    }
}

fn execute_gemm_bias_in_session(
    _op: &GemmBiasOp,
    session: &mut dyn BackendSession,
    _caches: &mut tenferro_runtime::ExtensionCacheStore,
    inputs: &[TensorRead<'_>],
) -> tenferro_tensor::Result<Vec<Tensor>> {
    if inputs.len() != 3 {
        return Err(tensor_error(
            "inputs",
            format!("expected 3 inputs, found {}", inputs.len()),
        ));
    }
    let x_shape = inputs[0].shape().to_vec();
    let weight_shape = inputs[1].shape().to_vec();
    if x_shape.len() != 2 || weight_shape.len() != 2 {
        return Err(tensor_error("shape", "gemm_bias expects two rank-2 inputs"));
    }
    let (in_dim, length) = (x_shape[0], x_shape[1]);
    let (weight_in, out_dim) = (weight_shape[0], weight_shape[1]);
    if in_dim != weight_in {
        return Err(tensor_error(
            "shape",
            format!("x has {in_dim} contracting rows but weight has {weight_in}"),
        ));
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
    let bias_data = match inputs[2].as_slice::<f32>() {
        Ok(data) => data,
        Err(_) => {
            bias_storage = session.to_contiguous_read(inputs[2].clone())?;
            bias_storage
                .as_slice::<f32>()
                .map_err(|error| tensor_error("bias", error.to_string()))?
        }
    };
    let mut y = vec![0.0f32; out_dim * length];
    cpu_kernels::input_mul_weight_transpose_bias_into(
        x_data,
        length,
        in_dim,
        weight_data,
        out_dim,
        bias_data,
        &mut y,
    );
    Ok(vec![Tensor::from_vec_col_major(vec![out_dim, length], y)?])
}

fn gemm_bias_session_supported<B: TensorBackend + 'static>(_op: &GemmBiasOp) -> bool {
    std::any::TypeId::of::<B>() == std::any::TypeId::of::<CpuBackend>()
}

define_extension_runtime! {
    runtime = GemmBiasRuntime,
    family_id = GEMM_BIAS_FAMILY_ID,
    op_type = GemmBiasOp,
    execute_in_session = execute_gemm_bias_in_session,
    session_supported = gemm_bias_session_supported,
    backend_bound = TensorBackend,
}

pub(crate) fn gemm_bias_extension_module(
    target: EagerExtensionTarget,
) -> tenferro_runtime::Result<Arc<dyn ExtensionModule>> {
    extension_module::<CpuBackend>(target.engine_id).map_err(|source| {
        tenferro_runtime::Error::runtime_state_source(
            "tenferro-ext::gemm_bias",
            ErrorPhase::Execution,
            source,
        )
    })
}

/// Eager-session method for the `gemm_bias` extension.
pub trait EagerSessionGemmBiasExt {
    /// Dense `y = weightᵀ · x + bias` for `f32` operands.
    fn gemm_bias(
        &mut self,
        x: &EagerTensor,
        weight: &EagerTensor,
        bias: &EagerTensor,
    ) -> tenferro_ad::Result<EagerTensor>;
}

impl EagerSessionGemmBiasExt for EagerSession<'_> {
    fn gemm_bias(
        &mut self,
        x: &EagerTensor,
        weight: &EagerTensor,
        bias: &EagerTensor,
    ) -> tenferro_ad::Result<EagerTensor> {
        let mut outputs = apply_eager_with_targeted_extension_in_session(
            self,
            Arc::new(GemmBiasOp),
            &[x, weight, bias],
            gemm_bias_extension_module,
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
