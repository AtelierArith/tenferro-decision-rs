//! Self-hosted eager CPU dense-`linear` extension op backed by `cpu-kernels`.
//!
//! `linear` contracts the last axis of `x` with the first axis of `weight`:
//! `y = x · weight`, `x (length, in)`, `weight (in, out)`, `y (length, out)`,
//! all column-major. The eager `dot_general` uses faer with per-call analysis
//! and allocation; this reaches [`cpu_kernels::matmul_col_major_into`].

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

/// Stable family identifier for the `linear` extension op.
pub const LINEAR_FAMILY_ID: &str = "tenferro-decision.linear.v1";

/// The dense `y = x · weight` extension op (two inputs, one output).
#[derive(Clone, Debug)]
pub struct LinearOp;

fn tensor_error(field: &'static str, message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::invalid_argument("tenferro-ext::linear", field, message)
}

impl ExtensionOp for LinearOp {
    fn family_id(&self) -> &'static str {
        LINEAR_FAMILY_ID
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<LinearOp>().is_some()
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
        let x_shape = ctx.input_shape(0)?;
        let weight_shape = ctx.input_shape(1)?;
        if x_shape.len() != 2 || weight_shape.len() != 2 {
            return Err(tensor_error("shape", "linear expects two rank-2 inputs"));
        }
        let (length, in_dim) = (x_shape[0].clone(), x_shape[1].clone());
        let (weight_in, out) = (weight_shape[0].clone(), weight_shape[1].clone());
        ctx.require_equal(in_dim, weight_in)?;
        Ok(vec![(ctx.input_dtype(0)?, vec![length, out])])
    }
}

fn execute_linear_in_session(
    _op: &LinearOp,
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
    let x_shape = inputs[0].shape().to_vec();
    let weight_shape = inputs[1].shape().to_vec();
    if x_shape.len() != 2 || weight_shape.len() != 2 {
        return Err(tensor_error("shape", "linear expects two rank-2 inputs"));
    }
    let (length, in_dim) = (x_shape[0], x_shape[1]);
    let (weight_in, out_dim) = (weight_shape[0], weight_shape[1]);
    if in_dim != weight_in {
        return Err(tensor_error(
            "shape",
            format!("x has {in_dim} contracted columns but weight has {weight_in} rows"),
        ));
    }
    let x_storage;
    let weight_storage;
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
    let mut y = vec![0.0f32; length * out_dim];
    cpu_kernels::matmul_col_major_into(x_data, weight_data, length, in_dim, out_dim, &mut y);
    Ok(vec![Tensor::from_vec_col_major(vec![length, out_dim], y)?])
}

fn linear_session_supported<B: TensorBackend + 'static>(_op: &LinearOp) -> bool {
    std::any::TypeId::of::<B>() == std::any::TypeId::of::<CpuBackend>()
}

define_extension_runtime! {
    runtime = LinearRuntime,
    family_id = LINEAR_FAMILY_ID,
    op_type = LinearOp,
    execute_in_session = execute_linear_in_session,
    session_supported = linear_session_supported,
    backend_bound = TensorBackend,
}

fn linear_extension_module(
    target: EagerExtensionTarget,
) -> tenferro_runtime::Result<Arc<dyn ExtensionModule>> {
    extension_module::<CpuBackend>(target.engine_id).map_err(|source| {
        tenferro_runtime::Error::runtime_state_source(
            "tenferro-ext::linear",
            ErrorPhase::Execution,
            source,
        )
    })
}

/// Eager-session method for the `linear` extension.
pub trait EagerSessionLinearExt {
    /// Dense `y = x · weight` for `f32` `x (length, in)` and `weight (in, out)`.
    fn linear(&mut self, x: &EagerTensor, weight: &EagerTensor)
    -> tenferro_ad::Result<EagerTensor>;
}

impl EagerSessionLinearExt for EagerSession<'_> {
    fn linear(
        &mut self,
        x: &EagerTensor,
        weight: &EagerTensor,
    ) -> tenferro_ad::Result<EagerTensor> {
        let mut outputs = apply_eager_with_targeted_extension_in_session(
            self,
            Arc::new(LinearOp),
            &[x, weight],
            linear_extension_module,
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

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_ad::EagerRuntime;

    fn lcg(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*state >> 40) as f32) / (1u64 << 24) as f32 - 0.5
    }

    #[test]
    fn linear_matches_naive() {
        let (length, in_dim, out_dim) = (4usize, 5usize, 3usize);
        let mut state = 13u64;
        let x: Vec<f32> = (0..length * in_dim).map(|_| lcg(&mut state)).collect();
        let w: Vec<f32> = (0..in_dim * out_dim).map(|_| lcg(&mut state)).collect();

        let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
        let got = runtime
            .with_eager_session(|session| {
                let x_t = session
                    .constant_from(Tensor::from_vec_col_major(vec![length, in_dim], x.clone())?)?;
                let w_t = session.constant_from(Tensor::from_vec_col_major(
                    vec![in_dim, out_dim],
                    w.clone(),
                )?)?;
                session.linear(&x_t, &w_t)
            })
            .unwrap()
            .unwrap()
            .value()
            .unwrap()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();

        let mut want = vec![0.0f32; length * out_dim];
        for t in 0..length {
            for o in 0..out_dim {
                let mut acc = 0.0f32;
                for i in 0..in_dim {
                    // x is column-major `(length, in)`; weight is row-major
                    // `(in, out)` storage.
                    acc += x[t + length * i] * w[i * out_dim + o];
                }
                want[t + length * o] = acc;
            }
        }
        let diff = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff < 1e-5, "linear diff {diff}: got {got:?} want {want:?}");
    }
}
