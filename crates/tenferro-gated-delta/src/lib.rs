//! Chunked Gated DeltaNet for the Jeff engine.
//!
//! Design: `docs/agents/specs/docs/12_TENFERRO_GATED_DELTA.md`. The crate owns
//! the causal depthwise convolution, the reference recurrent scan, the fused
//! host recurrent kernel, and the tenferro-backed chunked formulation.
//!
//! Implemented now:
//!
//! - [`ops`]: reference scalar/vector helpers
//! - [`conv`]: causal depthwise convolution with a fused SiLU
//! - [`reference`]: host recurrent scan (one value head)
//! - [`recurrent`]: fused host recurrent layer over a reusable workspace
//! - [`chunked`]: tenferro eager chunked scan (one value head)
//! - [`layer`]: full projection/normalization/scan/output wrapper
//! - [`plan`]: deterministic algorithm selection and prepared plans
//! - [`workspace`]: reusable execution buffers
//! - [`extension`]: the `GatedDelta` tenferro extension op
//! - [`gated_delta`]: the direct entry point dispatching on a plan
//!
//! The optional `cuda` feature exposes compiled raw CUDA kernels and scoped
//! layer requests (`cuda_request`); `jeff-infer`'s CUDA forward runs its
//! DeltaNet layers through `CudaRequest::layer_time_first` (hardware-validated
//! on an RTX 3060).

pub mod chunked;
pub mod config;
pub mod conv;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod cuda_chunked_layer;
#[cfg(feature = "cuda")]
pub mod cuda_layer;
#[cfg(feature = "cuda")]
pub mod cuda_request;
pub mod extension;
pub mod layer;
pub mod ops;
pub mod plan;
#[cfg(feature = "cuda")]
pub mod raw;
pub mod recurrent;
pub mod reference;
pub mod tensor_layer;
pub mod traced_layer;
pub mod workspace;

pub use config::{Algorithm, GatedDeltaConfig};
pub use conv::{causal_depthwise_silu, causal_depthwise_silu_into};
pub use extension::{EagerSessionGatedDeltaExt, GATED_DELTA_FAMILY_ID, GatedDeltaOp};
pub use layer::{
    GatedDeltaWeightSlices, GatedDeltaWeights, delta_layer_reference, delta_layer_tenferro,
};
pub use plan::{AlgorithmChoice, BackendCaps, GatedDeltaPlan, resolve_algorithm};
pub use recurrent::{delta_layer_recurrent, delta_layer_recurrent_slices};
pub use reference::{DeltaScanInputs, delta_scan_reference};
pub use tensor_layer::{
    GatedDeltaKernelWeights, GatedDeltaTensorWeights, NativeDeltaConstants, ProjectedDeltaTensors,
    delta_layer_from_projected, delta_layer_from_projected_with_constants,
    delta_layer_tenferro_cached, delta_layer_tenferro_native, delta_layer_tenferro_prepared_mask,
    prepare_kernel_weights, prepare_native_constants, prepare_tensor_weights,
};
pub use workspace::GatedDeltaWorkspace;

use decision_core::{DecisionError, Result};
use tenferro_ad::EagerSession;

/// Direct entry point: run the formulation frozen in `plan`.
///
/// `Chunked` uses the eager `session` (tenferro-backed projections and scan);
/// `Reference` and `Recurrent` are host-only and ignore the session. The
/// recurrent path writes through `ws` and returns an owned copy.
pub fn gated_delta(
    session: &mut EagerSession<'_>,
    cfg: &GatedDeltaConfig,
    weights: &GatedDeltaWeights,
    x: &[f32],
    mask: &[f32],
    plan: &GatedDeltaPlan,
    ws: &mut GatedDeltaWorkspace,
) -> tenferro_ad::Result<Vec<f32>> {
    match plan.algorithm() {
        Algorithm::Reference => {
            delta_layer_reference(cfg, weights, x, mask).map_err(layer::invalid_weights)
        }
        Algorithm::Recurrent => delta_layer_recurrent(cfg, weights, x, mask, ws)
            .map(|output| output.to_vec())
            .map_err(layer::invalid_weights),
        Algorithm::Chunked => delta_layer_tenferro(session, cfg, weights, x, mask),
    }
}

/// Eager convenience wrapper over [`gated_delta`] without an explicit plan.
pub fn gated_delta_eager(
    session: &mut EagerSession<'_>,
    cfg: &GatedDeltaConfig,
    weights: &GatedDeltaWeights,
    x: &[f32],
    mask: &[f32],
    ws: &mut GatedDeltaWorkspace,
) -> tenferro_ad::Result<Vec<f32>> {
    let plan = GatedDeltaPlan::from_config(cfg, mask.len());
    gated_delta(session, cfg, weights, x, mask, &plan, ws)
}

/// Build the `n x n` row-major effective matrix `M = I + L`.
///
/// `beta[i]` is the write strength, `keys[i]` the (already L2-normalized) key
/// vector of row `i`, and `cum_decay[i]` the cumulative log-decay `c_i`.
pub fn build_effective_matrix(
    beta: &[f32],
    keys: &[Vec<f32>],
    cum_decay: &[f32],
) -> Result<Vec<Vec<f32>>> {
    let n = beta.len();
    if keys.len() != n || cum_decay.len() != n {
        return Err(DecisionError::invalid_field(
            "delta",
            "beta, keys, and cum_decay must have the same length",
        ));
    }
    let key_dim = keys.first().map(Vec::len).unwrap_or(0);
    if keys.iter().any(|k| k.len() != key_dim) {
        return Err(DecisionError::invalid_field(
            "delta.keys",
            "all key vectors must have the same length",
        ));
    }

    let mut m = vec![vec![0.0f32; n]; n];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for i in 0..n {
        for j in 0..i {
            let dot: f32 = keys[i].iter().zip(&keys[j]).map(|(a, b)| a * b).sum();
            m[i][j] = beta[i] * dot * (cum_decay[i] - cum_decay[j]).exp();
        }
    }
    Ok(m)
}

/// Solve `M x = rhs` for a unit-diagonal lower-triangular row-major `M`.
///
/// `rhs` is `n x dv`, row-major. The system matrix must have ones on the
/// diagonal (as produced by [`build_effective_matrix`]).
pub fn forward_substitute(m: &[Vec<f32>], rhs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
    let n = m.len();
    if rhs.len() != n {
        return Err(DecisionError::invalid_field(
            "delta",
            "matrix and right-hand side must agree on the chunk length",
        ));
    }
    let dv = rhs.first().map(Vec::len).unwrap_or(0);
    if rhs.iter().any(|row| row.len() != dv) {
        return Err(DecisionError::invalid_field(
            "delta.rhs",
            "all right-hand-side rows must have the same length",
        ));
    }
    for (i, row) in m.iter().enumerate() {
        if row.len() != n {
            return Err(DecisionError::invalid_field(
                "delta.matrix",
                "matrix must be square",
            ));
        }
        if row[i] != 1.0 {
            return Err(DecisionError::invalid_field(
                "delta.matrix",
                "the diagonal must be unit",
            ));
        }
    }

    let mut out = vec![vec![0.0f32; dv]; n];
    for i in 0..n {
        for d in 0..dv {
            let mut value = rhs[i][d];
            for j in 0..i {
                value -= m[i][j] * out[j][d];
            }
            out[i][d] = value;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_when_no_coupling() {
        let beta = [1.0f32, 1.0];
        let keys = vec![vec![1.0f32, 0.0], vec![0.0, 1.0]];
        let cum_decay = [0.0f32, 0.0];
        let m = build_effective_matrix(&beta, &keys, &cum_decay).unwrap();
        assert_eq!(m, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);

        let rhs = vec![vec![2.0f32], vec![3.0]];
        let x = forward_substitute(&m, &rhs).unwrap();
        assert_eq!(x, rhs);
    }

    #[test]
    fn solves_lower_triangular_system() {
        let m = vec![vec![1.0f32, 0.0], vec![0.5, 1.0]];
        let rhs = vec![vec![1.0f32], vec![0.0]];
        let x = forward_substitute(&m, &rhs).unwrap();
        assert_eq!(x, vec![vec![1.0f32], vec![-0.5]]);
    }

    #[test]
    fn rejects_shape_mismatch() {
        assert!(build_effective_matrix(&[1.0], &[], &[0.0]).is_err());
        assert!(forward_substitute(&[vec![1.0, 0.0]], &[vec![1.0]]).is_err());
    }
}
