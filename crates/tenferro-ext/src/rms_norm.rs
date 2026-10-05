//! Self-hosted eager CPU feature-last RMSNorm extension op backed by
//! `cpu-kernels`.
//!
//! The eager `norm::rms_norm` is ~10 composed ops (square-reduce, scale, add
//! eps, rsqrt, broadcast, two multiplies, ...). This fuses it into one pass for
//! a `(length, width)` activation normalized over the last axis.

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

/// Stable family identifier for the feature-last RMSNorm extension op.
pub const RMS_NORM_FAMILY_ID: &str = "tenferro-decision.rms_norm_last.v1";

/// Feature-last RMSNorm (`centered` ⇒ scale `1 + weight`).
#[derive(Clone, Debug)]
pub struct RmsNormLastOp {
    /// Whether the effective scale is `1 + weight` (`true`) or `weight`.
    pub centered: bool,
    /// `eps` as `f32` bits.
    pub eps_bits: u32,
}

fn tensor_error(field: &'static str, message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::invalid_argument("tenferro-ext::rms_norm_last", field, message)
}

impl ExtensionOp for RmsNormLastOp {
    fn family_id(&self) -> &'static str {
        RMS_NORM_FAMILY_ID
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_u8(self.centered as u8);
        hasher.write_u32(self.eps_bits);
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<RmsNormLastOp>()
            .is_some_and(|other| other.centered == self.centered && other.eps_bits == self.eps_bits)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        2
    }

    fn output_count(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        ctx: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        let shape = ctx.input_shape(0)?;
        if shape.len() != 2 {
            return Err(tensor_error("shape", "rms_norm_last expects (rows, width)"));
        }
        Ok(vec![(ctx.input_dtype(0)?, shape.to_vec())])
    }
}

fn execute_rms_norm_in_session(
    op: &RmsNormLastOp,
    session: &mut dyn BackendSession,
    _caches: &mut tenferro_runtime::ExtensionCacheStore,
    inputs: &[TensorRead<'_>],
) -> tenferro_tensor::Result<Vec<Tensor>> {
    if inputs.len() != 2 {
        return Err(tensor_error(
            "inputs",
            format!("expected 2 inputs, found {}", inputs.len()),
        ));
    }
    let shape = inputs[0].shape().to_vec();
    if shape.len() != 2 {
        return Err(tensor_error("shape", "rms_norm_last expects (rows, width)"));
    }
    let (rows, width) = (shape[0], shape[1]);

    let x_storage;
    let x = match inputs[0].as_slice::<f32>() {
        Ok(data) => data,
        Err(_) => {
            x_storage = session.to_contiguous_read(inputs[0].clone())?;
            x_storage
                .as_slice::<f32>()
                .map_err(|error| tensor_error("x", error.to_string()))?
        }
    };
    let weight_storage;
    let weight = match inputs[1].as_slice::<f32>() {
        Ok(data) => data,
        Err(_) => {
            weight_storage = session.to_contiguous_read(inputs[1].clone())?;
            weight_storage
                .as_slice::<f32>()
                .map_err(|error| tensor_error("weight", error.to_string()))?
        }
    };
    if weight.len() != width {
        return Err(tensor_error("shape", "weight must be (width)"));
    }

    let mut out = vec![0.0f32; rows * width];
    cpu_kernels::rms_norm_last_into(
        x,
        rows,
        width,
        weight,
        op.centered,
        f32::from_bits(op.eps_bits),
        &mut out,
    );
    Ok(vec![Tensor::from_vec_col_major(shape, out)?])
}

fn rms_norm_session_supported<B: TensorBackend + 'static>(_op: &RmsNormLastOp) -> bool {
    std::any::TypeId::of::<B>() == std::any::TypeId::of::<CpuBackend>()
}

define_extension_runtime! {
    runtime = RmsNormLastRuntime,
    family_id = RMS_NORM_FAMILY_ID,
    op_type = RmsNormLastOp,
    execute_in_session = execute_rms_norm_in_session,
    session_supported = rms_norm_session_supported,
    backend_bound = TensorBackend,
}

fn rms_norm_extension_module(
    target: EagerExtensionTarget,
) -> tenferro_runtime::Result<Arc<dyn ExtensionModule>> {
    extension_module::<CpuBackend>(target.engine_id).map_err(|source| {
        tenferro_runtime::Error::runtime_state_source(
            "tenferro-ext::rms_norm_last",
            ErrorPhase::Execution,
            source,
        )
    })
}

/// Eager-session method for the feature-last RMSNorm extension.
pub trait EagerSessionRmsNormExt {
    /// RMSNorm over the last axis of a `(rows, width)` `f32` activation.
    fn rms_norm_last(
        &mut self,
        x: &EagerTensor,
        weight: &EagerTensor,
        centered: bool,
        eps: f64,
    ) -> tenferro_ad::Result<EagerTensor>;
}

impl EagerSessionRmsNormExt for EagerSession<'_> {
    fn rms_norm_last(
        &mut self,
        x: &EagerTensor,
        weight: &EagerTensor,
        centered: bool,
        eps: f64,
    ) -> tenferro_ad::Result<EagerTensor> {
        let op = RmsNormLastOp {
            centered,
            eps_bits: (eps as f32).to_bits(),
        };
        let mut outputs = apply_eager_with_targeted_extension_in_session(
            self,
            Arc::new(op),
            &[x, weight],
            rms_norm_extension_module,
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
