//! Self-hosted eager CPU ModernBERT RoPE extension op backed by `cpu-kernels`.
//!
//! `tenferro_infer::rope::rope_modernbert` builds a `(length, head/2)` cos/sin
//! table on the host and then runs slice/broadcast/mul/sub/add/concatenate as
//! separate session ops. This fuses the whole thing into one op backed by
//! [`cpu_kernels::rope_modernbert_into`].

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

/// Stable family identifier for the ModernBERT RoPE extension op.
pub const ROPE_FAMILY_ID: &str = "tenferro-decision.rope_modernbert.v1";

/// ModernBERT RoPE over `(B, H, L, head_dim)`.
#[derive(Clone, Debug)]
pub struct RopeOp {
    /// RoPE `base` stored as `f32` bits so the payload hash/eq see it.
    pub base_bits: u32,
}

fn tensor_error(field: &'static str, message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::invalid_argument("tenferro-ext::rope_modernbert", field, message)
}

impl ExtensionOp for RopeOp {
    fn family_id(&self) -> &'static str {
        ROPE_FAMILY_ID
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_u32(self.base_bits);
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<RopeOp>()
            .is_some_and(|other| other.base_bits == self.base_bits)
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
        if shape.len() != 4 {
            return Err(tensor_error("shape", "rope_modernbert expects rank 4"));
        }
        Ok(vec![(ctx.input_dtype(0)?, shape.to_vec())])
    }
}

fn execute_rope_in_session(
    op: &RopeOp,
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
    if shape.len() != 4 {
        return Err(tensor_error("shape", "rope_modernbert expects rank 4"));
    }
    let (batch, heads, length, head_dim) = (shape[0], shape[1], shape[2], shape[3]);

    let storage;
    let x = match inputs[0].as_slice::<f32>() {
        Ok(data) => data,
        Err(_) => {
            storage = session.to_contiguous_read(inputs[0].clone())?;
            storage
                .as_slice::<f32>()
                .map_err(|error| tensor_error("x", error.to_string()))?
        }
    };
    let mut out = vec![0.0f32; batch * heads * length * head_dim];
    cpu_kernels::rope_modernbert_into(
        x,
        f32::from_bits(op.base_bits),
        batch,
        heads,
        length,
        head_dim,
        &mut out,
    );
    Ok(vec![Tensor::from_vec_col_major(shape, out)?])
}

fn rope_session_supported<B: TensorBackend + 'static>(_op: &RopeOp) -> bool {
    std::any::TypeId::of::<B>() == std::any::TypeId::of::<CpuBackend>()
}

define_extension_runtime! {
    runtime = RopeRuntime,
    family_id = ROPE_FAMILY_ID,
    op_type = RopeOp,
    execute_in_session = execute_rope_in_session,
    session_supported = rope_session_supported,
    backend_bound = TensorBackend,
}

fn rope_extension_module(
    target: EagerExtensionTarget,
) -> tenferro_runtime::Result<Arc<dyn ExtensionModule>> {
    extension_module::<CpuBackend>(target.engine_id).map_err(|source| {
        tenferro_runtime::Error::runtime_state_source(
            "tenferro-ext::rope_modernbert",
            ErrorPhase::Execution,
            source,
        )
    })
}

/// Eager-session method for the ModernBERT RoPE extension.
pub trait EagerSessionRopeExt {
    /// ModernBERT RoPE on a `(B, H, L, head_dim)` `f32` activation.
    fn rope_modernbert(&mut self, x: &EagerTensor, base: f64) -> tenferro_ad::Result<EagerTensor>;
}

impl EagerSessionRopeExt for EagerSession<'_> {
    fn rope_modernbert(&mut self, x: &EagerTensor, base: f64) -> tenferro_ad::Result<EagerTensor> {
        let op = RopeOp {
            base_bits: (base as f32).to_bits(),
        };
        let mut outputs = apply_eager_with_targeted_extension_in_session(
            self,
            Arc::new(op),
            &[x],
            rope_extension_module,
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
