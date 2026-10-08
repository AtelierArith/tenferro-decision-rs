//! Self-hosted eager CPU GEMM extension op backed by `cpu-kernels`.
//!
//! The eager `dot_general` path pays per-call overhead (analysis, provider
//! entry, and output allocation) on top of the same-class GEMM kernel. This
//! extension exposes a single dense projection as a first-class tenferro op so
//! the eager graph can reach our BLAS-class host kernel
//! (`cpu-kernels::input_mul_weight_transpose`, which routes large `f32` GEMMs
//! through Apple Accelerate's `cblas_sgemm` on macOS and `matrixmultiply`
//! elsewhere) without a separate eager dispatch.
//!
//! Semantics: `y = weightᵀ · x` with
//!
//! * `x` — logical shape `(in, length)`,
//! * `weight` — logical shape `(in, out)` (the engines' canonical `(in, out)`
//!   layout),
//! * `y` — logical shape `(out, length)`.
//!
//! All operands are `f32`. tenferro tensors are column-major, which lines up
//! with [`cpu_kernels::input_mul_weight_transpose`]: reading the column-major
//! `(in, length)` `x` as row-major `(length, in)` and the column-major
//! `(in, out)` `weight` as row-major `(out, in)`, the kernel's row-major
//! `(length, out)` result is exactly the column-major buffer of the logical
//! `(out, length)` output — no explicit transposes are needed.

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

/// Stable family identifier for the `gemm` extension op.
pub const GEMM_FAMILY_ID: &str = "tenferro-decision.gemm.v1";

/// The dense `y = weightᵀ · x` extension op (two inputs, one output).
#[derive(Clone, Debug)]
pub struct GemmOp;

fn tensor_error(field: &'static str, message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::invalid_argument("tenferro-ext::gemm", field, message)
}

impl ExtensionOp for GemmOp {
    fn family_id(&self) -> &'static str {
        GEMM_FAMILY_ID
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<GemmOp>().is_some()
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
            return Err(tensor_error("shape", "gemm expects two rank-2 inputs"));
        }
        let (x_in, length) = (x_shape[0].clone(), x_shape[1].clone());
        let (weight_in, out) = (weight_shape[0].clone(), weight_shape[1].clone());
        ctx.require_equal(x_in, weight_in)?;
        Ok(vec![(ctx.input_dtype(0)?, vec![out, length])])
    }
}

fn execute_gemm_in_session(
    _op: &GemmOp,
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
        return Err(tensor_error("shape", "gemm expects two rank-2 inputs"));
    }
    let (in_dim, length) = (x_shape[0], x_shape[1]);
    let (weight_in, out_dim) = (weight_shape[0], weight_shape[1]);
    if in_dim != weight_in {
        return Err(tensor_error(
            "shape",
            format!("x has {in_dim} contracting rows but weight has {weight_in}"),
        ));
    }
    // Prefer borrowing host slices directly — the eager path passes compact
    // tensors, and `to_contiguous_read` would deep-copy them every call. Fall
    // back to materialization only for strided views.
    let x_storage;
    let weight_storage;
    let (x_data, weight_data) = match (inputs[0].as_slice::<f32>(), inputs[1].as_slice::<f32>()) {
        (Ok(x_data), Ok(weight_data)) => (x_data, weight_data),
        _ => {
            x_storage = session.to_contiguous_read(inputs[0].clone())?;
            weight_storage = session.to_contiguous_read(inputs[1].clone())?;
            let x_data = x_storage
                .as_slice::<f32>()
                .map_err(|error| tensor_error("x", error.to_string()))?;
            let weight_data = weight_storage
                .as_slice::<f32>()
                .map_err(|error| tensor_error("weight", error.to_string()))?;
            (x_data, weight_data)
        }
    };
    // Row-major `(length, out)` result == column-major `(out, length)` output.
    let y = cpu_kernels::input_mul_weight_transpose(x_data, length, in_dim, weight_data, out_dim);
    Ok(vec![Tensor::from_vec_col_major(vec![out_dim, length], y)?])
}

fn gemm_session_supported<B: TensorBackend + 'static>(_op: &GemmOp) -> bool {
    std::any::TypeId::of::<B>() == std::any::TypeId::of::<CpuBackend>()
}

define_extension_runtime! {
    runtime = GemmRuntime,
    family_id = GEMM_FAMILY_ID,
    op_type = GemmOp,
    execute_in_session = execute_gemm_in_session,
    session_supported = gemm_session_supported,
    backend_bound = TensorBackend,
}

fn gemm_extension_module(
    target: EagerExtensionTarget,
) -> tenferro_runtime::Result<Arc<dyn ExtensionModule>> {
    // This extension is CPU-only; model callers select native dot_general
    // on other backends.
    extension_module::<CpuBackend>(target.engine_id).map_err(|source| {
        tenferro_runtime::Error::runtime_state_source(
            "tenferro-ext::gemm",
            ErrorPhase::Execution,
            source,
        )
    })
}

/// Eager-session methods for the `gemm` extension.
pub trait EagerSessionGemmExt {
    /// Dense `y = weightᵀ · x` for `f32` operands.
    ///
    /// `x` has shape `(in, length)` and `weight` has shape `(in, out)`; the
    /// result has shape `(out, length)`.
    fn gemm(&mut self, x: &EagerTensor, weight: &EagerTensor) -> tenferro_ad::Result<EagerTensor>;
}

impl EagerSessionGemmExt for EagerSession<'_> {
    fn gemm(&mut self, x: &EagerTensor, weight: &EagerTensor) -> tenferro_ad::Result<EagerTensor> {
        one_output(apply_eager_with_targeted_extension_in_session(
            self,
            Arc::new(GemmOp),
            &[x, weight],
            gemm_extension_module,
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
    fn gemm_matches_naive_loop() {
        let (in_dim, out_dim, length) = (5usize, 3usize, 4usize);
        let mut state = 42u64;
        // Column-major logical `x (in, length)` and `weight (in, out)`.
        let x_data: Vec<f32> = (0..in_dim * length).map(|_| lcg(&mut state)).collect();
        let weight_data: Vec<f32> = (0..in_dim * out_dim).map(|_| lcg(&mut state)).collect();

        let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
        let output = runtime
            .with_eager_session(|session| {
                let x = session.constant_from(Tensor::from_vec_col_major(
                    vec![in_dim, length],
                    x_data.clone(),
                )?)?;
                let weight = session.constant_from(Tensor::from_vec_col_major(
                    vec![in_dim, out_dim],
                    weight_data.clone(),
                )?)?;
                session.gemm(&x, &weight)
            })
            .unwrap()
            .unwrap();
        let got = output.value().unwrap().as_slice::<f32>().unwrap().to_vec();

        // `y[o, t] = Σ_i weight[i, o] · x[i, t]`, stored column-major `(out, length)`.
        let mut want = vec![0.0f32; out_dim * length];
        for o in 0..out_dim {
            for t in 0..length {
                let mut acc = 0.0f32;
                for i in 0..in_dim {
                    acc += weight_data[i + o * in_dim] * x_data[i + t * in_dim];
                }
                want[o + t * out_dim] = acc;
            }
        }
        let diff = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff < 1e-5, "gemm diff {diff}: got {got:?} want {want:?}");
    }
}
