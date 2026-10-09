//! Self-hosted eager CPU fused Laya attention block extension op.
//!
//! Replaces the `reshape → split → head reshape → transpose → RoPE ×2 →
//! attention` chain (~20 eager ops) with one op backed by
//! [`cpu_kernels::laya_attention_block_into`]. The whole computation runs in the
//! host-friendly feature-first `(d, L, B)` layout (per-head `head_dim`
//! contiguous), so no transposes are needed.

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

/// Stable family identifier for the fused Laya attention block extension op.
pub const LAYA_ATTENTION_BLOCK_FAMILY_ID: &str = "tenferro-decision.laya_attention_block.v1";

/// Fused split + RoPE + masked attention for a fused `(3d, L, B)` projection.
#[derive(Clone, Debug)]
pub struct LayaAttentionBlockOp {
    /// Hidden width `d`.
    pub hidden: usize,
    /// Number of heads.
    pub heads: usize,
    /// RoPE base as `f32` bits; `0` disables RoPE.
    pub rope_base_bits: u32,
}

fn tensor_error(field: &'static str, message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::invalid_argument("tenferro-ext::laya_attention_block", field, message)
}

impl ExtensionOp for LayaAttentionBlockOp {
    fn family_id(&self) -> &'static str {
        LAYA_ATTENTION_BLOCK_FAMILY_ID
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_usize(self.hidden);
        hasher.write_usize(self.heads);
        hasher.write_u32(self.rope_base_bits);
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<LayaAttentionBlockOp>()
            .is_some_and(|other| {
                other.hidden == self.hidden
                    && other.heads == self.heads
                    && other.rope_base_bits == self.rope_base_bits
            })
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
        if shape.len() != 3 {
            return Err(tensor_error("shape", "attention block expects (3d, L, B)"));
        }
        Ok(vec![(
            ctx.input_dtype(0)?,
            vec![
                SymDim::from(self.hidden),
                shape[1].clone(),
                shape[2].clone(),
            ],
        )])
    }
}

fn execute_laya_attention_block_in_session(
    op: &LayaAttentionBlockOp,
    session: &mut dyn BackendSession,
    caches: &mut tenferro_runtime::ExtensionCacheStore,
    inputs: &[TensorRead<'_>],
) -> tenferro_tensor::Result<Vec<Tensor>> {
    if inputs.len() != 2 {
        return Err(tensor_error(
            "inputs",
            format!("expected 2 inputs, found {}", inputs.len()),
        ));
    }
    let shape = inputs[0].shape().to_vec();
    if shape.len() != 3 || shape[0] != 3 * op.hidden {
        return Err(tensor_error("shape", "attention block expects (3d, L, B)"));
    }
    let length = shape[1];
    let batch = shape[2];

    let qkv_storage;
    let qkv = match inputs[0].as_slice::<f32>() {
        Ok(data) => data,
        Err(_) => {
            qkv_storage = session.to_contiguous_read(inputs[0].clone())?;
            qkv_storage
                .as_slice::<f32>()
                .map_err(|error| tensor_error("qkv", error.to_string()))?
        }
    };
    let keep_storage;
    let keep = match inputs[1].as_slice::<bool>() {
        Ok(data) => data,
        Err(_) => {
            keep_storage = session.to_contiguous_read(inputs[1].clone())?;
            keep_storage
                .as_slice::<bool>()
                .map_err(|error| tensor_error("keep", error.to_string()))?
        }
    };
    if keep.len() != length * length * batch {
        return Err(tensor_error("keep", "keep must be (L, L, B)"));
    }
    let mut out = vec![0.0f32; op.hidden * length * batch];
    let key = tenferro_runtime::ExtensionCacheKey::new(
        LAYA_ATTENTION_BLOCK_FAMILY_ID,
        "head_workspaces",
        u64::from(op.rope_base_bits),
    );
    if caches
        .get::<cpu_kernels::LayaAttentionWorkspace>(&key)
        .is_none()
    {
        caches.put_with_retained_bytes(
            key,
            cpu_kernels::LayaAttentionWorkspace::default(),
            cpu_kernels::LayaAttentionWorkspace::retained_bytes,
        );
    }
    let workspace = caches
        .get_mut::<cpu_kernels::LayaAttentionWorkspace>(&key)
        .ok_or_else(|| tensor_error("cache", "attention workspace unavailable"))?;
    cpu_kernels::laya_attention_block_with_workspace(
        qkv,
        keep,
        op.hidden,
        op.heads,
        length,
        batch,
        f32::from_bits(op.rope_base_bits),
        &mut out,
        workspace,
    );

    Ok(vec![Tensor::from_vec_col_major(
        vec![op.hidden, length, batch],
        out,
    )?])
}

fn laya_attention_block_session_supported<B: TensorBackend + 'static>(
    _op: &LayaAttentionBlockOp,
) -> bool {
    std::any::TypeId::of::<B>() == std::any::TypeId::of::<CpuBackend>()
}

define_extension_runtime! {
    runtime = LayaAttentionBlockRuntime,
    family_id = LAYA_ATTENTION_BLOCK_FAMILY_ID,
    op_type = LayaAttentionBlockOp,
    execute_in_session = execute_laya_attention_block_in_session,
    session_supported = laya_attention_block_session_supported,
    backend_bound = TensorBackend,
}

fn laya_attention_block_extension_module(
    target: EagerExtensionTarget,
) -> tenferro_runtime::Result<Arc<dyn ExtensionModule>> {
    extension_module::<CpuBackend>(target.engine_id).map_err(|source| {
        tenferro_runtime::Error::runtime_state_source(
            "tenferro-ext::laya_attention_block",
            ErrorPhase::Execution,
            source,
        )
    })
}

/// Eager-session method for the fused Laya attention block.
pub trait EagerSessionLayaAttentionExt {
    /// Fused split + RoPE + masked attention for a `(3d, L, B)` `f32`
    /// projection, returning `(d, L, B)`. `rope_base == 0.0` disables RoPE.
    #[allow(clippy::too_many_arguments)]
    fn laya_attention_block(
        &mut self,
        qkv: &EagerTensor,
        keep: &EagerTensor,
        hidden: usize,
        heads: usize,
        rope_base: f64,
    ) -> tenferro_ad::Result<EagerTensor>;
}

impl EagerSessionLayaAttentionExt for EagerSession<'_> {
    fn laya_attention_block(
        &mut self,
        qkv: &EagerTensor,
        keep: &EagerTensor,
        hidden: usize,
        heads: usize,
        rope_base: f64,
    ) -> tenferro_ad::Result<EagerTensor> {
        let op = LayaAttentionBlockOp {
            hidden,
            heads,
            rope_base_bits: (rope_base as f32).to_bits(),
        };
        let mut outputs = apply_eager_with_targeted_extension_in_session(
            self,
            Arc::new(op),
            &[qkv, keep],
            laya_attention_block_extension_module,
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
