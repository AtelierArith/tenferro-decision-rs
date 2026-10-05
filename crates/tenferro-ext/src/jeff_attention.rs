//! Self-hosted eager CPU fused Jeff full-attention extension op.
//!
//! Replaces the `project_heads (reshape+transpose) → RMSNorm ×2 → partial RoPE
//! ×2 → mask → attention → gate → merge` chain with one op backed by
//! `cpu-kernels::jeff_full_attention_into`. The `q`/`k`/`v`/`gate` projections
//! stay as separate `linear` ops.

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

/// Stable family identifier for the fused Jeff full-attention extension op.
pub const JEFF_FULL_ATTENTION_FAMILY_ID: &str = "tenferro-decision.jeff_full_attention.v1";

/// Fused post-projection Jeff full attention.
#[derive(Clone, Debug)]
pub struct JeffFullAttentionOp {
    /// Number of heads.
    pub heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// RoPE channels (partial rotary).
    pub rotary_dim: usize,
    /// RoPE `theta` as `f32` bits.
    pub theta_bits: u32,
    /// RMSNorm `eps` as `f32` bits.
    pub eps_bits: u32,
}

fn tensor_error(field: &'static str, message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::invalid_argument("tenferro-ext::jeff_full_attention", field, message)
}

impl ExtensionOp for JeffFullAttentionOp {
    fn family_id(&self) -> &'static str {
        JEFF_FULL_ATTENTION_FAMILY_ID
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_usize(self.heads);
        hasher.write_usize(self.head_dim);
        hasher.write_usize(self.rotary_dim);
        hasher.write_u32(self.theta_bits);
        hasher.write_u32(self.eps_bits);
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<JeffFullAttentionOp>()
            .is_some_and(|other| {
                other.heads == self.heads
                    && other.head_dim == self.head_dim
                    && other.rotary_dim == self.rotary_dim
                    && other.theta_bits == self.theta_bits
                    && other.eps_bits == self.eps_bits
            })
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        7
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
            return Err(tensor_error(
                "shape",
                "full attention expects (length, width)",
            ));
        }
        Ok(vec![(ctx.input_dtype(0)?, shape.to_vec())])
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_jeff_full_attention_in_session(
    op: &JeffFullAttentionOp,
    session: &mut dyn BackendSession,
    _caches: &mut tenferro_runtime::ExtensionCacheStore,
    inputs: &[TensorRead<'_>],
) -> tenferro_tensor::Result<Vec<Tensor>> {
    if inputs.len() != 7 {
        return Err(tensor_error(
            "inputs",
            format!("expected 7 inputs, found {}", inputs.len()),
        ));
    }
    let shape = inputs[0].shape().to_vec();
    if shape.len() != 2 {
        return Err(tensor_error(
            "shape",
            "full attention expects (length, width)",
        ));
    }
    let (length, width) = (shape[0], shape[1]);
    if width != op.heads * op.head_dim {
        return Err(tensor_error("shape", "width must equal heads * head_dim"));
    }

    // Materialize any strided operand once, then take all slices.
    let owned: Vec<Option<Tensor>> = inputs
        .iter()
        .map(|input| {
            if input.as_slice::<f32>().is_ok() {
                Ok(None)
            } else {
                session.to_contiguous_read(input.clone()).map(Some)
            }
        })
        .collect::<tenferro_tensor::Result<_>>()?;
    let mut reads: Vec<&[f32]> = Vec::with_capacity(7);
    for (index, (input, owned)) in inputs.iter().zip(&owned).enumerate() {
        let name = match index {
            0 => "q",
            1 => "k",
            2 => "v",
            3 => "gate",
            4 => "q_norm",
            5 => "k_norm",
            _ => "mask",
        };
        let slice = match owned {
            Some(tensor) => tensor.as_slice::<f32>(),
            None => input.as_slice::<f32>(),
        }
        .map_err(|error| tensor_error(name, error.to_string()))?;
        reads.push(slice);
    }
    let (q, k, v, gate, q_norm, k_norm, mask) = (
        reads[0], reads[1], reads[2], reads[3], reads[4], reads[5], reads[6],
    );
    if q_norm.len() != op.head_dim || k_norm.len() != op.head_dim || mask.len() != length {
        return Err(tensor_error(
            "shape",
            "q_norm/k_norm must be (head_dim) and mask (length)",
        ));
    }

    let mut out = vec![0.0f32; length * width];
    cpu_kernels::jeff_full_attention_into(
        q,
        k,
        v,
        gate,
        q_norm,
        k_norm,
        mask,
        length,
        op.heads,
        op.head_dim,
        op.rotary_dim,
        f32::from_bits(op.theta_bits),
        f32::from_bits(op.eps_bits),
        &mut out,
    );
    Ok(vec![Tensor::from_vec_col_major(shape, out)?])
}

fn jeff_full_attention_session_supported<B: TensorBackend + 'static>(
    _op: &JeffFullAttentionOp,
) -> bool {
    std::any::TypeId::of::<B>() == std::any::TypeId::of::<CpuBackend>()
}

define_extension_runtime! {
    runtime = JeffFullAttentionRuntime,
    family_id = JEFF_FULL_ATTENTION_FAMILY_ID,
    op_type = JeffFullAttentionOp,
    execute_in_session = execute_jeff_full_attention_in_session,
    session_supported = jeff_full_attention_session_supported,
    backend_bound = TensorBackend,
}

fn jeff_full_attention_extension_module(
    target: EagerExtensionTarget,
) -> tenferro_runtime::Result<Arc<dyn ExtensionModule>> {
    extension_module::<CpuBackend>(target.engine_id).map_err(|source| {
        tenferro_runtime::Error::runtime_state_source(
            "tenferro-ext::jeff_full_attention",
            ErrorPhase::Execution,
            source,
        )
    })
}

/// Eager-session method for the fused Jeff full-attention op.
pub trait EagerSessionJeffAttentionExt {
    /// Fused post-projection full attention: `q/k/v/gate` are `(length, width)`,
    /// `q_norm`/`k_norm` are `(head_dim)`, `mask` is the `(length)` active-key
    /// mask; returns the merged `(length, width)` activation.
    #[allow(clippy::too_many_arguments)]
    fn jeff_full_attention(
        &mut self,
        q: &EagerTensor,
        k: &EagerTensor,
        v: &EagerTensor,
        gate: &EagerTensor,
        q_norm: &EagerTensor,
        k_norm: &EagerTensor,
        mask: &EagerTensor,
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        theta: f64,
        eps: f64,
    ) -> tenferro_ad::Result<EagerTensor>;
}

impl EagerSessionJeffAttentionExt for EagerSession<'_> {
    #[allow(clippy::too_many_arguments)]
    fn jeff_full_attention(
        &mut self,
        q: &EagerTensor,
        k: &EagerTensor,
        v: &EagerTensor,
        gate: &EagerTensor,
        q_norm: &EagerTensor,
        k_norm: &EagerTensor,
        mask: &EagerTensor,
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        theta: f64,
        eps: f64,
    ) -> tenferro_ad::Result<EagerTensor> {
        let op = JeffFullAttentionOp {
            heads,
            head_dim,
            rotary_dim,
            theta_bits: (theta as f32).to_bits(),
            eps_bits: (eps as f32).to_bits(),
        };
        let mut outputs = apply_eager_with_targeted_extension_in_session(
            self,
            Arc::new(op),
            &[q, k, v, gate, q_norm, k_norm, mask],
            jeff_full_attention_extension_module,
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
