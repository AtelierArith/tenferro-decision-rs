//! Self-hosted tenferro extension ops for the decision engines.
//!
//! tenferro has no elementwise `erf`, so this crate adds one as a first-class
//! extension op — the same mechanism `tenferro-linalg`/`tenferro-fft` use — and
//! exposes it together with the exact (erf-based) GELU on the eager session.
//! CPU sessions use the fused extension; other backends compose the same erf
//! approximation coefficients from native elementwise operations.
//!
//! The `f32` kernel ports the MLX Metal `erff`/`expm1f` sequence used by Laya's
//! reference (`extern/Laya.jl/src/mathfns.jl`), so the eager GELU matches the
//! host reference bit-for-bit in `f32`. `f64` uses Abramowitz–Stegun 7.1.26.
//!
//! [`gemm`] adds a dense `y = weightᵀ · x` op backed by the shared
//! `cpu-kernels` GEMM, so the eager session can reach our BLAS-class host path.

#![allow(clippy::approx_constant, clippy::excessive_precision)]

mod gated_silu;
mod geglu;
mod gemm;
mod gemm_bias;
mod jeff_attention;
mod laya_attention;
mod layernorm;
mod linear;
mod native_erf;
mod rms_norm;

pub use gated_silu::{EagerSessionGatedSiluExt, GATED_SILU_FAMILY_ID, GatedSiluOp};
pub use geglu::{EagerSessionGegluExt, GEGLU_FAMILY_ID, GegluOp};
pub use gemm::{EagerSessionGemmExt, GEMM_FAMILY_ID, GemmOp};
pub use gemm_bias::{EagerSessionGemmBiasExt, GEMM_BIAS_FAMILY_ID, GemmBiasOp};
pub use jeff_attention::{
    EagerSessionJeffAttentionExt, JEFF_FULL_ATTENTION_FAMILY_ID, JeffFullAttentionOp,
};
pub use laya_attention::{
    EagerSessionLayaAttentionExt, LAYA_ATTENTION_BLOCK_FAMILY_ID, LayaAttentionBlockOp,
};
pub use layernorm::{EagerSessionLayerNormExt, LAYER_NORM_FF_FAMILY_ID, LayerNormFeatureFirstOp};
pub use linear::{EagerSessionLinearExt, LINEAR_FAMILY_ID, LinearOp};
pub use native_erf::{erf_tensor_native, gelu_erf_tensor_native};
pub use rms_norm::{EagerSessionRmsNormExt, RMS_NORM_FAMILY_ID, RmsNormLastOp};

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

/// Whether this admitted eager session can execute this crate's CPU extensions.
///
/// Check the backend session, rather than tensor dtype: an F32 tensor may be
/// resident on a GPU. This only borrows the CPU execution marker and does not
/// transfer tensors or change the backend.
pub fn cpu_extensions_supported(session: &mut EagerSession<'_>) -> bool {
    tenferro_cpu::with_cpu_exec_session(session.backend_session(), |_| ()).is_some()
}

/// Stable family identifier for the `erf` extension op.
pub const ERF_FAMILY_ID: &str = "tenferro-decision.erf.v1";

/// The elementwise `erf` extension op (one input, one output, same shape/dtype).
#[derive(Clone, Debug)]
pub struct ErfOp;

impl ExtensionOp for ErfOp {
    fn family_id(&self) -> &'static str {
        ERF_FAMILY_ID
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<ErfOp>().is_some()
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
        Ok(vec![(ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec())])
    }
}

fn execute_erf_in_session(
    _op: &ErfOp,
    _session: &mut dyn BackendSession,
    _caches: &mut tenferro_runtime::ExtensionCacheStore,
    inputs: &[TensorRead<'_>],
) -> tenferro_tensor::Result<Vec<Tensor>> {
    if inputs.len() != 1 {
        return Err(tenferro_tensor::Error::invalid_argument(
            "tenferro-ext::erf",
            "inputs",
            format!("expected 1 input, found {}", inputs.len()),
        ));
    }
    let input = &inputs[0];
    let shape = input.shape().to_vec();
    let output = match input.dtype() {
        DType::F32 => {
            let data = input.as_slice::<f32>()?;
            Tensor::from_vec_col_major(
                shape,
                data.iter()
                    .map(|value| cpu_kernels::erf_f32(*value))
                    .collect(),
            )?
        }
        DType::F64 => {
            let data = input.as_slice::<f64>()?;
            Tensor::from_vec_col_major(shape, data.iter().map(|value| erf_f64(*value)).collect())?
        }
        other => {
            return Err(tenferro_tensor::Error::invalid_argument(
                "tenferro-ext::erf",
                "dtype",
                format!("unsupported dtype {other:?}"),
            ));
        }
    };
    Ok(vec![output])
}

fn erf_session_supported<B: TensorBackend + 'static>(_op: &ErfOp) -> bool {
    std::any::TypeId::of::<B>() == std::any::TypeId::of::<CpuBackend>()
}

define_extension_runtime! {
    runtime = ErfRuntime,
    family_id = ERF_FAMILY_ID,
    op_type = ErfOp,
    execute_in_session = execute_erf_in_session,
    session_supported = erf_session_supported,
    backend_bound = TensorBackend,
}

fn erf_extension_module(
    target: EagerExtensionTarget,
) -> tenferro_runtime::Result<Arc<dyn ExtensionModule>> {
    // This extension has a CPU implementation only. Unsupported backends
    // report an extension error; they never download tensors implicitly.
    extension_module::<CpuBackend>(target.engine_id).map_err(|source| {
        tenferro_runtime::Error::runtime_state_source(
            "tenferro-ext::erf",
            ErrorPhase::Execution,
            source,
        )
    })
}

/// Eager-session methods for the `erf` extension and the exact GELU.
pub trait EagerSessionErfExt {
    /// Elementwise `erf`; CPU extension or native composition by backend.
    fn erf(&mut self, x: &EagerTensor) -> tenferro_ad::Result<EagerTensor>;

    /// Exact (erf-based) GELU: `x * (1 + erf(x / sqrt(2))) / 2`.
    fn gelu_erf(&mut self, x: &EagerTensor) -> tenferro_ad::Result<EagerTensor>;
}

impl EagerSessionErfExt for EagerSession<'_> {
    fn erf(&mut self, x: &EagerTensor) -> tenferro_ad::Result<EagerTensor> {
        if !cpu_extensions_supported(self) {
            return erf_tensor_native(self, x);
        }
        one_output(
            apply_eager_with_targeted_extension_in_session(
                self,
                Arc::new(ErfOp),
                &[x],
                erf_extension_module,
            )?,
            "erf",
        )
    }

    fn gelu_erf(&mut self, x: &EagerTensor) -> tenferro_ad::Result<EagerTensor> {
        if !cpu_extensions_supported(self) {
            return gelu_erf_tensor_native(self, x);
        }
        let scaled = self.scale_real(x, std::f64::consts::FRAC_1_SQRT_2)?;
        let erf = self.erf(&scaled)?;
        let one = scalar_like(self, x, 1.0)?;
        let gate = self.add(&erf, &one)?;
        let half = self.scale_real(x, 0.5)?;
        self.mul(&half, &gate)
    }
}

fn one_output(outputs: Vec<EagerTensor>, op: &'static str) -> tenferro_ad::Result<EagerTensor> {
    let mut outputs = outputs.into_iter();
    match (outputs.next(), outputs.next()) {
        (Some(value), None) => Ok(value),
        _ => Err(tenferro_ad::Error::TensorRuntime(
            tenferro_tensor::Error::invalid_argument(
                "tenferro-ext",
                op,
                "extension returned an unexpected number of outputs",
            ),
        )),
    }
}

fn scalar_like(
    session: &mut EagerSession<'_>,
    reference: &EagerTensor,
    value: f64,
) -> tenferro_ad::Result<EagerTensor> {
    let tensor = match reference.dtype() {
        DType::F32 => Tensor::from_vec_col_major(vec![], vec![value as f32])?,
        DType::F64 => Tensor::from_vec_col_major(vec![], vec![value])?,
        other => {
            return Err(tenferro_ad::Error::TensorRuntime(
                tenferro_tensor::Error::invalid_argument(
                    "tenferro-ext",
                    "dtype",
                    format!("unsupported dtype {other:?}"),
                ),
            ));
        }
    };
    session.constant_from_host(tensor)
}

/// Abramowitz–Stegun 7.1.26 (`|ε| ≤ 1.5e-7`) for `f64`.
fn erf_f64(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = ((((1.061_405_429 * t - 1.453_152_027) * t + 1.421_413_741) * t - 0.284_496_736)
        * t
        + 0.254_829_592)
        * t;
    sign * (1.0 - poly * (-x * x).exp())
}
