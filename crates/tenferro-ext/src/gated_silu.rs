//! Self-hosted eager CPU gated-SiLU extension op backed by `cpu-kernels`.
//!
//! `activation::gated_silu` composes `silu(gate) * up` from neg/exp/add/div/mul
//! (~5 ops); this fuses it into one elementwise pass.

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

/// Stable family identifier for the gated-SiLU extension op.
pub const GATED_SILU_FAMILY_ID: &str = "tenferro-decision.gated_silu.v1";

/// `silu(gate) * up`, elementwise.
#[derive(Clone, Debug)]
pub struct GatedSiluOp;

fn tensor_error(field: &'static str, message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::invalid_argument("tenferro-ext::gated_silu", field, message)
}

impl ExtensionOp for GatedSiluOp {
    fn family_id(&self) -> &'static str {
        GATED_SILU_FAMILY_ID
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<GatedSiluOp>().is_some()
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
        Ok(vec![(ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec())])
    }
}

fn execute_gated_silu_in_session(
    _op: &GatedSiluOp,
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
    if inputs[1].shape() != shape.as_slice() {
        return Err(tensor_error("shape", "gate and up must share a shape"));
    }
    let gate_storage;
    let gate = match inputs[0].as_slice::<f32>() {
        Ok(data) => data,
        Err(_) => {
            gate_storage = session.to_contiguous_read(inputs[0].clone())?;
            gate_storage
                .as_slice::<f32>()
                .map_err(|error| tensor_error("gate", error.to_string()))?
        }
    };
    let up_storage;
    let up = match inputs[1].as_slice::<f32>() {
        Ok(data) => data,
        Err(_) => {
            up_storage = session.to_contiguous_read(inputs[1].clone())?;
            up_storage
                .as_slice::<f32>()
                .map_err(|error| tensor_error("up", error.to_string()))?
        }
    };
    let mut out = vec![0.0f32; gate.len()];
    cpu_kernels::gated_silu_into(gate, up, &mut out);
    Ok(vec![Tensor::from_vec_col_major(shape, out)?])
}

fn gated_silu_session_supported<B: TensorBackend + 'static>(_op: &GatedSiluOp) -> bool {
    std::any::TypeId::of::<B>() == std::any::TypeId::of::<CpuBackend>()
}

define_extension_runtime! {
    runtime = GatedSiluRuntime,
    family_id = GATED_SILU_FAMILY_ID,
    op_type = GatedSiluOp,
    execute_in_session = execute_gated_silu_in_session,
    session_supported = gated_silu_session_supported,
    backend_bound = TensorBackend,
}

fn gated_silu_extension_module(
    target: EagerExtensionTarget,
) -> tenferro_runtime::Result<Arc<dyn ExtensionModule>> {
    extension_module::<CpuBackend>(target.engine_id).map_err(|source| {
        tenferro_runtime::Error::runtime_state_source(
            "tenferro-ext::gated_silu",
            ErrorPhase::Execution,
            source,
        )
    })
}

/// Eager-session method for the gated-SiLU extension.
pub trait EagerSessionGatedSiluExt {
    /// `silu(gate) * up`, elementwise.
    fn gated_silu(
        &mut self,
        gate: &EagerTensor,
        up: &EagerTensor,
    ) -> tenferro_ad::Result<EagerTensor>;
}

impl EagerSessionGatedSiluExt for EagerSession<'_> {
    fn gated_silu(
        &mut self,
        gate: &EagerTensor,
        up: &EagerTensor,
    ) -> tenferro_ad::Result<EagerTensor> {
        let mut outputs = apply_eager_with_targeted_extension_in_session(
            self,
            Arc::new(GatedSiluOp),
            &[gate, up],
            gated_silu_extension_module,
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
