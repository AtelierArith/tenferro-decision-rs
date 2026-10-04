//! `GatedDelta` tenferro extension op.
//!
//! Design: `docs/agents/specs/docs/12_TENFERRO_GATED_DELTA.md` §12. The op
//! exposes the Gated DeltaNet layer to the eager/traced graphs through the same
//! `ExtensionOp` / `define_extension_runtime!` mechanism as `tenferro-linalg`
//! and the self-hosted `erf` op. Descriptor fields are semantic only; `x`,
//! `mask`, and the nine weight tensors are inputs in the fixed order below:
//!
//! `x (L, hidden)`, `mask (L,)`, `qkv (2k+v, hidden)`, `z (v, hidden)`,
//! `a (value_heads, hidden)`, `b (value_heads, hidden)`,
//! `conv (2k+v, conv_taps)`, `a_decay (value_heads,)`, `dt_bias (value_heads,)`,
//! `norm (value_dim,)`, `out_proj (hidden, v)`.
//!
//! Each 2-D input is column-major, so its buffer is exactly the fused host
//! kernel's row-major operand (`x` is row-major `(hidden, L)`, `qkv` is
//! row-major `(hidden, 2k+v)`, ...). Execution therefore reads every input in
//! place and runs the fused host recurrent kernel on the CPU; no hidden transfer
//! or process-global cache is involved. The output is `(hidden, L)` column-major.

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

use crate::config::{Algorithm, GatedDeltaConfig};
use crate::layer::GatedDeltaWeightSlices;
use crate::recurrent::delta_layer_recurrent_slices;
use crate::workspace::GatedDeltaWorkspace;

/// Stable family identifier for the `GatedDelta` extension op.
pub const GATED_DELTA_FAMILY_ID: &str = "tenferro-gated-delta.gated_delta.v1";

/// Number of inputs consumed by [`GatedDeltaOp`].
pub const GATED_DELTA_INPUT_COUNT: usize = 11;

/// Semantic descriptor for one `GatedDelta` invocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GatedDeltaOp {
    /// Model hidden width.
    pub hidden: usize,
    /// Number of key (query) heads.
    pub key_heads: usize,
    /// Number of value heads.
    pub value_heads: usize,
    /// Key/query head width.
    pub key_dim: usize,
    /// Value head width.
    pub value_dim: usize,
    /// Causal convolution kernel size.
    pub conv_taps: usize,
    /// RMSNorm epsilon, stored as raw bits for `Eq`/hashing.
    pub eps_bits: u32,
    /// Chunk size (semantic: part of the op identity).
    pub chunk_size: usize,
    /// Formulation used by the direct path; the op itself always runs the fused
    /// host kernel, so this only documents intent.
    pub algorithm: Algorithm,
}

impl GatedDeltaOp {
    /// Build the descriptor from a validated config.
    pub fn from_config(config: &GatedDeltaConfig) -> Self {
        Self {
            hidden: config.hidden,
            key_heads: config.key_heads,
            value_heads: config.value_heads,
            key_dim: config.key_dim,
            value_dim: config.value_dim,
            conv_taps: config.conv_taps,
            eps_bits: config.eps.to_bits(),
            chunk_size: config.chunk_size,
            algorithm: config.algorithm,
        }
    }

    /// Reconstruct the layer config.
    pub fn config(&self) -> GatedDeltaConfig {
        GatedDeltaConfig {
            hidden: self.hidden,
            key_heads: self.key_heads,
            value_heads: self.value_heads,
            key_dim: self.key_dim,
            value_dim: self.value_dim,
            conv_taps: self.conv_taps,
            eps: f32::from_bits(self.eps_bits),
            chunk_size: self.chunk_size,
            algorithm: self.algorithm,
        }
    }
}

impl ExtensionOp for GatedDeltaOp {
    fn family_id(&self) -> &'static str {
        GATED_DELTA_FAMILY_ID
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_usize(self.hidden);
        hasher.write_usize(self.key_heads);
        hasher.write_usize(self.value_heads);
        hasher.write_usize(self.key_dim);
        hasher.write_usize(self.value_dim);
        hasher.write_usize(self.conv_taps);
        hasher.write_u32(self.eps_bits);
        hasher.write_usize(self.chunk_size);
        hasher.write_u8(self.algorithm as u8);
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<GatedDeltaOp>()
            .is_some_and(|other| self == other)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        GATED_DELTA_INPUT_COUNT
    }

    fn output_count(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        ctx: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        // `x` is `(L, hidden)`; the layer returns `(hidden, L)`.
        let shape = ctx.input_shape(0)?.to_vec();
        if shape.len() != 2 {
            return Err(tensor_error("x", "expected a 2-D `x`"));
        }
        let out = vec![shape[1].clone(), shape[0].clone()];
        Ok(vec![(ctx.input_dtype(0)?, out)])
    }
}

fn tensor_error(field: &'static str, message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::invalid_argument("tenferro-gated-delta::gated_delta", field, message)
}

fn execute_gated_delta_in_session(
    op: &GatedDeltaOp,
    _session: &mut dyn BackendSession,
    _caches: &mut tenferro_runtime::ExtensionCacheStore,
    inputs: &[TensorRead<'_>],
) -> tenferro_tensor::Result<Vec<Tensor>> {
    if inputs.len() != GATED_DELTA_INPUT_COUNT {
        return Err(tensor_error(
            "inputs",
            format!(
                "expected {GATED_DELTA_INPUT_COUNT} inputs, found {}",
                inputs.len()
            ),
        ));
    }

    // Every input is read in place: the column-major buffers are already the
    // fused kernel's row-major operands, so no transpose or copy is needed.
    let slice = |index: usize, field: &'static str| -> tenferro_tensor::Result<&[f32]> {
        inputs[index]
            .as_slice::<f32>()
            .map_err(|error| tensor_error(field, error.to_string()))
    };

    let x_shape = inputs[0].shape();
    if x_shape.len() != 2 || x_shape[1] != op.hidden {
        return Err(tensor_error(
            "x",
            "expected a 2-D `x` whose second dimension is `hidden`",
        ));
    }
    let length = x_shape[0];
    if inputs[1].shape() != [length] {
        return Err(tensor_error(
            "mask",
            "length must equal `x`'s first dimension",
        ));
    }

    let weight_slices = GatedDeltaWeightSlices {
        qkv: slice(2, "qkv")?,
        z: slice(3, "z")?,
        a: slice(4, "a")?,
        b: slice(5, "b")?,
        conv: slice(6, "conv")?,
        a_decay: slice(7, "a_decay")?,
        dt_bias: slice(8, "dt_bias")?,
        norm: slice(9, "norm")?,
        out_proj: slice(10, "out_proj")?,
    };
    let config = op.config();
    weight_slices
        .validate(&config)
        .map_err(|error| tensor_error("weights", error.to_string()))?;

    let mut workspace = GatedDeltaWorkspace::new();
    let output = delta_layer_recurrent_slices(
        &config,
        &weight_slices,
        slice(0, "x")?,
        slice(1, "mask")?,
        &mut workspace,
    )
    .map_err(|error| tensor_error("weights", error.to_string()))?;

    // The kernel output is row-major `(hidden, length)`; tensors are
    // column-major, so transpose into the `(hidden, length)` output buffer.
    let mut column_major = vec![0.0f32; op.hidden * length];
    for row in 0..op.hidden {
        for column in 0..length {
            column_major[row + column * op.hidden] = output[row * length + column];
        }
    }
    Ok(vec![Tensor::from_vec_col_major(
        vec![op.hidden, length],
        column_major,
    )?])
}

fn gated_delta_session_supported<B: TensorBackend + 'static>(_op: &GatedDeltaOp) -> bool {
    std::any::TypeId::of::<B>() == std::any::TypeId::of::<CpuBackend>()
}

define_extension_runtime! {
    runtime = GatedDeltaRuntime,
    family_id = GATED_DELTA_FAMILY_ID,
    op_type = GatedDeltaOp,
    execute_in_session = execute_gated_delta_in_session,
    session_supported = gated_delta_session_supported,
    backend_bound = TensorBackend,
}

fn gated_delta_extension_module(
    target: EagerExtensionTarget,
) -> tenferro_runtime::Result<Arc<dyn ExtensionModule>> {
    extension_module::<CpuBackend>(target.engine_id).map_err(|source| {
        tenferro_runtime::Error::runtime_state_source(
            "tenferro-gated-delta::gated_delta",
            ErrorPhase::Execution,
            source,
        )
    })
}

/// Eager-session methods for the `GatedDelta` extension.
pub trait EagerSessionGatedDeltaExt {
    /// Apply `GatedDelta` with `x`, `mask`, and the nine weight tensors.
    fn gated_delta(
        &mut self,
        op: GatedDeltaOp,
        inputs: &[&EagerTensor],
    ) -> tenferro_ad::Result<EagerTensor>;
}

impl EagerSessionGatedDeltaExt for EagerSession<'_> {
    fn gated_delta(
        &mut self,
        op: GatedDeltaOp,
        inputs: &[&EagerTensor],
    ) -> tenferro_ad::Result<EagerTensor> {
        let outputs = apply_eager_with_targeted_extension_in_session(
            self,
            Arc::new(op),
            inputs,
            gated_delta_extension_module,
        )?;
        one_output(outputs)
    }
}

fn one_output(outputs: Vec<EagerTensor>) -> tenferro_ad::Result<EagerTensor> {
    let mut outputs = outputs.into_iter();
    match (outputs.next(), outputs.next()) {
        (Some(value), None) => Ok(value),
        _ => Err(tenferro_ad::Error::TensorRuntime(tensor_error(
            "outputs",
            "extension returned an unexpected number of outputs",
        ))),
    }
}
