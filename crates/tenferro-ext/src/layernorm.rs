//! Self-hosted eager CPU feature-first LayerNorm extension op backed by
//! `cpu-kernels`.
//!
//! The eager `norm::layer_norm` works on the *last* axis, so a feature-first
//! activation `(d, ...)` makes the eager path run `transpose → layer_norm →
//! transpose` and a handful of elementwise/reduction ops per norm. This
//! extension exposes the whole operation as one first-class op backed by
//! [`cpu_kernels::layer_norm_feature_first_into`], matching `layer_norm_host`.
//!
//! Semantics: normalize over axis 0 of a column-major `(d, ...)` activation,
//! then scale by a `(d,)` weight and add an optional `(d,)` bias.

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

/// Stable family identifier for the feature-first LayerNorm extension op.
pub const LAYER_NORM_FF_FAMILY_ID: &str = "tenferro-decision.layernorm_ff.v1";

/// Feature-first LayerNorm (two inputs, or three with a bias).
#[derive(Clone, Debug)]
pub struct LayerNormFeatureFirstOp {
    /// `eps` stored as `f32` bits so the payload hash/eq see it.
    pub eps_bits: u32,
    /// Whether the third input is a `(d,)` bias.
    pub with_bias: bool,
}

fn tensor_error(field: &'static str, message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::invalid_argument("tenferro-ext::layernorm_ff", field, message)
}

impl ExtensionOp for LayerNormFeatureFirstOp {
    fn family_id(&self) -> &'static str {
        LAYER_NORM_FF_FAMILY_ID
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_u32(self.eps_bits);
        hasher.write_u8(self.with_bias as u8);
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<LayerNormFeatureFirstOp>()
            .is_some_and(|other| {
                other.eps_bits == self.eps_bits && other.with_bias == self.with_bias
            })
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        if self.with_bias { 3 } else { 2 }
    }

    fn output_count(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        ctx: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        let shape = ctx.input_shape(0)?;
        if shape.len() < 2 {
            return Err(tensor_error(
                "shape",
                "layernorm expects a rank >= 2 activation",
            ));
        }
        Ok(vec![(ctx.input_dtype(0)?, shape.to_vec())])
    }
}

fn execute_layernorm_in_session(
    op: &LayerNormFeatureFirstOp,
    session: &mut dyn BackendSession,
    _caches: &mut tenferro_runtime::ExtensionCacheStore,
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
    let shape = inputs[0].shape().to_vec();
    if shape.len() < 2 {
        return Err(tensor_error(
            "shape",
            "layernorm expects a rank >= 2 activation",
        ));
    }
    let d = shape[0];
    let cols: usize = shape[1..].iter().product();

    // Prefer borrowing compact host slices; materialize only strided views.
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
    let bias_data = if op.with_bias {
        match inputs[2].as_slice::<f32>() {
            Ok(data) => Some(data),
            Err(_) => {
                bias_storage = session.to_contiguous_read(inputs[2].clone())?;
                Some(
                    bias_storage
                        .as_slice::<f32>()
                        .map_err(|error| tensor_error("bias", error.to_string()))?,
                )
            }
        }
    } else {
        None
    };

    let mut y = vec![0.0f32; d * cols];
    cpu_kernels::layer_norm_feature_first_into(
        x_data,
        d,
        cols,
        weight_data,
        bias_data,
        f32::from_bits(op.eps_bits),
        &mut y,
    );
    Ok(vec![Tensor::from_vec_col_major(shape, y)?])
}

fn layernorm_session_supported<B: TensorBackend + 'static>(_op: &LayerNormFeatureFirstOp) -> bool {
    std::any::TypeId::of::<B>() == std::any::TypeId::of::<CpuBackend>()
}

define_extension_runtime! {
    runtime = LayerNormFeatureFirstRuntime,
    family_id = LAYER_NORM_FF_FAMILY_ID,
    op_type = LayerNormFeatureFirstOp,
    execute_in_session = execute_layernorm_in_session,
    session_supported = layernorm_session_supported,
    backend_bound = TensorBackend,
}

fn layernorm_extension_module(
    target: EagerExtensionTarget,
) -> tenferro_runtime::Result<Arc<dyn ExtensionModule>> {
    extension_module::<CpuBackend>(target.engine_id).map_err(|source| {
        tenferro_runtime::Error::runtime_state_source(
            "tenferro-ext::layernorm_ff",
            ErrorPhase::Execution,
            source,
        )
    })
}

/// Eager-session method for the feature-first LayerNorm extension.
pub trait EagerSessionLayerNormExt {
    /// LayerNorm over axis 0 of `x` (`(d, ...)`), with `(d,)` weight and
    /// optional `(d,)` bias.
    fn layer_norm_feature_first(
        &mut self,
        x: &EagerTensor,
        weight: &EagerTensor,
        bias: Option<&EagerTensor>,
        eps: f32,
    ) -> tenferro_ad::Result<EagerTensor>;
}

impl EagerSessionLayerNormExt for EagerSession<'_> {
    fn layer_norm_feature_first(
        &mut self,
        x: &EagerTensor,
        weight: &EagerTensor,
        bias: Option<&EagerTensor>,
        eps: f32,
    ) -> tenferro_ad::Result<EagerTensor> {
        let op = LayerNormFeatureFirstOp {
            eps_bits: eps.to_bits(),
            with_bias: bias.is_some(),
        };
        let mut inputs: Vec<&EagerTensor> = vec![x, weight];
        if let Some(bias) = bias {
            inputs.push(bias);
        }
        one_output(apply_eager_with_targeted_extension_in_session(
            self,
            Arc::new(op),
            &inputs,
            layernorm_extension_module,
        )?)
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

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_ad::EagerRuntime;

    fn lcg(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*state >> 40) as f32) / (1u64 << 24) as f32 - 0.5
    }

    #[test]
    fn layernorm_ff_matches_naive() {
        let (d, l, b) = (5usize, 3usize, 2usize);
        let eps = 1e-5f32;
        let mut state = 7u64;
        let x: Vec<f32> = (0..d * l * b).map(|_| lcg(&mut state)).collect();
        let weight: Vec<f32> = (0..d).map(|_| lcg(&mut state)).collect();
        let bias: Vec<f32> = (0..d).map(|_| lcg(&mut state)).collect();

        let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
        let run = |bias_in: Option<Vec<f32>>| -> Vec<f32> {
            runtime
                .with_eager_session(|session| {
                    let x_t = session
                        .constant_from(Tensor::from_vec_col_major(vec![d, l, b], x.clone())?)?;
                    let w_t = session
                        .constant_from(Tensor::from_vec_col_major(vec![d], weight.clone())?)?;
                    let b_t =
                        match &bias_in {
                            Some(bias) => Some(session.constant_from(
                                Tensor::from_vec_col_major(vec![d], bias.clone())?,
                            )?),
                            None => None,
                        };
                    session.layer_norm_feature_first(&x_t, &w_t, b_t.as_ref(), eps)
                })
                .unwrap()
                .unwrap()
                .value()
                .unwrap()
                .as_slice::<f32>()
                .unwrap()
                .to_vec()
        };

        for bias_in in [None, Some(bias.clone())] {
            let got = run(bias_in.clone());
            let mut want = vec![0.0f32; d * l * b];
            for c in 0..l * b {
                let col = &x[c * d..(c + 1) * d];
                let mean = col.iter().sum::<f32>() / d as f32;
                let var = col.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
                let inv = 1.0 / (var + eps).sqrt();
                for i in 0..d {
                    let mut value = (col[i] - mean) * inv * weight[i];
                    if let Some(bias) = &bias_in {
                        value += bias[i];
                    }
                    want[c * d + i] = value;
                }
            }
            let diff = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(diff < 1e-5, "layernorm diff {diff}");
        }
    }
}
