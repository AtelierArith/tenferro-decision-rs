//! Self-hosted eager CPU GeGLU extension op backed by `cpu-kernels`.
//!
//! Fuses Laya's MLP activation `slice(value) + slice(gate) + gelu_erf(value) +
//! mul` into one op, using the shared exact (erf-based) GELU
//! ([`cpu_kernels::geglu_into`]).

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

/// Stable family identifier for the GeGLU extension op.
pub const GEGLU_FAMILY_ID: &str = "tenferro-decision.geglu.v1";

/// GeGLU (`gelu(first half) * second half`) over a feature-first activation.
#[derive(Clone, Debug)]
pub struct GegluOp {
    /// Half the leading (feature) extent.
    pub intermediate: usize,
}

fn tensor_error(field: &'static str, message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::invalid_argument("tenferro-ext::geglu", field, message)
}

impl ExtensionOp for GegluOp {
    fn family_id(&self) -> &'static str {
        GEGLU_FAMILY_ID
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_usize(self.intermediate);
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<GegluOp>()
            .is_some_and(|other| other.intermediate == self.intermediate)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        ctx: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        let shape = ctx.input_shape(0)?;
        if shape.is_empty() {
            return Err(tensor_error(
                "shape",
                "geglu expects a rank >= 1 activation",
            ));
        }
        let mut out = shape.to_vec();
        out[0] = SymDim::from(self.intermediate);
        Ok(vec![(ctx.input_dtype(0)?, out)])
    }
}

fn execute_geglu_in_session(
    op: &GegluOp,
    session: &mut dyn BackendSession,
    _caches: &mut tenferro_runtime::ExtensionCacheStore,
    inputs: &[TensorRead<'_>],
) -> tenferro_tensor::Result<Vec<Tensor>> {
    if inputs.len() != 1 {
        return Err(tensor_error(
            "inputs",
            format!("expected 1 input, found {}", inputs.len()),
        ));
    }
    let shape = inputs[0].shape().to_vec();
    if shape.is_empty() || shape[0] != 2 * op.intermediate {
        return Err(tensor_error(
            "shape",
            "geglu input leading extent must be 2 * intermediate",
        ));
    }
    let cols: usize = shape[1..].iter().product();

    let storage;
    let u = match inputs[0].as_slice::<f32>() {
        Ok(data) => data,
        Err(_) => {
            storage = session.to_contiguous_read(inputs[0].clone())?;
            storage
                .as_slice::<f32>()
                .map_err(|error| tensor_error("u", error.to_string()))?
        }
    };
    let mut out = vec![0.0f32; op.intermediate * cols];
    cpu_kernels::geglu_into(u, op.intermediate, cols, &mut out);
    let mut out_shape = shape;
    out_shape[0] = op.intermediate;
    Ok(vec![Tensor::from_vec_col_major(out_shape, out)?])
}

fn geglu_session_supported<B: TensorBackend + 'static>(_op: &GegluOp) -> bool {
    std::any::TypeId::of::<B>() == std::any::TypeId::of::<CpuBackend>()
}

define_extension_runtime! {
    runtime = GegluRuntime,
    family_id = GEGLU_FAMILY_ID,
    op_type = GegluOp,
    execute_in_session = execute_geglu_in_session,
    session_supported = geglu_session_supported,
    backend_bound = TensorBackend,
}

fn geglu_extension_module(
    target: EagerExtensionTarget,
) -> tenferro_runtime::Result<Arc<dyn ExtensionModule>> {
    extension_module::<CpuBackend>(target.engine_id).map_err(|source| {
        tenferro_runtime::Error::runtime_state_source(
            "tenferro-ext::geglu",
            ErrorPhase::Execution,
            source,
        )
    })
}

/// Eager-session method for the GeGLU extension.
pub trait EagerSessionGegluExt {
    /// `gelu(first half) * second half` over the leading (feature) axis of a
    /// `f32` activation whose leading extent is `2 * intermediate`.
    fn geglu(&mut self, u: &EagerTensor, intermediate: usize) -> tenferro_ad::Result<EagerTensor>;
}

impl EagerSessionGegluExt for EagerSession<'_> {
    fn geglu(&mut self, u: &EagerTensor, intermediate: usize) -> tenferro_ad::Result<EagerTensor> {
        let mut outputs = apply_eager_with_targeted_extension_in_session(
            self,
            Arc::new(GegluOp { intermediate }),
            &[u],
            geglu_extension_module,
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
