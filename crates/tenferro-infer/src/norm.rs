//! LayerNorm and RMSNorm.
//!
//! Both operate over the last axis of `x` and broadcast their `(d,)` weight
//! (and optional bias) back over the leading axes.

use tenferro_ad::{EagerSession, EagerTensor, Result};

use crate::util::{broadcast_axis, broadcast_last_vector, scalar_like};

/// LayerNorm over the last axis: `(x - mean) / sqrt(var + eps) * weight + bias`.
pub fn layer_norm(
    session: &mut EagerSession<'_>,
    x: &EagerTensor,
    weight: &EagerTensor,
    bias: Option<&EagerTensor>,
    eps: f64,
) -> Result<EagerTensor> {
    let rank = x.shape().len();
    let axis = rank - 1;
    let width = x.shape()[axis] as f64;
    let target = x.shape().to_vec();

    let sum = session.reduce_sum(x, Some(&[axis]))?;
    let mean = crate::util::scale_real(session, &sum, 1.0 / width)?;
    let mean = broadcast_axis(session, &mean, &target, axis)?;

    let centered = session.sub(x, &mean)?;
    let sq = session.reduce_sum_squares(&centered, &[axis])?;
    let var = crate::util::scale_real(session, &sq, 1.0 / width)?;
    let eps = scalar_like(session, x, eps)?;
    let var_eps = session.add(&var, &eps)?;
    let inv = session.rsqrt(&var_eps)?;
    let inv = broadcast_axis(session, &inv, &target, axis)?;

    let normed = session.mul(&centered, &inv)?;
    let weight = broadcast_last_vector(session, weight, &target)?;
    let scaled = session.mul(&normed, &weight)?;
    match bias {
        Some(bias) => {
            let bias = broadcast_last_vector(session, bias, &target)?;
            session.add(&scaled, &bias)
        }
        None => Ok(scaled),
    }
}

/// RMSNorm over the last axis.
///
/// When `centered` is `true` the effective scale is `1 + weight`; otherwise it
/// is `weight`. The `eps` is added inside the square root, matching the
/// reference.
pub fn rms_norm(
    session: &mut EagerSession<'_>,
    x: &EagerTensor,
    weight: &EagerTensor,
    centered: bool,
    eps: f64,
) -> Result<EagerTensor> {
    let eps = scalar_like(session, x, eps)?;
    rms_norm_with_epsilon(session, x, weight, centered, &eps)
}

/// RMSNorm with a caller-prepared rank-zero epsilon tensor.
///
/// Reuse epsilon across chunks or forwards sharing the same eager runtime to
/// avoid repeated host registration/upload. Its dtype must match `x`.
pub fn rms_norm_with_epsilon(
    session: &mut EagerSession<'_>,
    x: &EagerTensor,
    weight: &EagerTensor,
    centered: bool,
    eps: &EagerTensor,
) -> Result<EagerTensor> {
    if x.shape().is_empty() || !eps.shape().is_empty() || eps.dtype() != x.dtype() {
        return Err(tenferro_ad::Error::TensorRuntime(
            tenferro_tensor::Error::invalid_argument(
                "rms_norm_with_epsilon",
                "shape/dtype",
                "x must have a feature axis and epsilon must be a matching-dtype scalar",
            ),
        ));
    }
    let rank = x.shape().len();
    let axis = rank - 1;
    let width = x.shape()[axis] as f64;
    let target = x.shape().to_vec();

    let sq = session.reduce_sum_squares(x, &[axis])?;
    let mean_sq = crate::util::scale_real(session, &sq, 1.0 / width)?;
    let denom = session.add(&mean_sq, eps)?;
    let inv = session.rsqrt(&denom)?;
    let inv = broadcast_axis(session, &inv, &target, axis)?;
    let normed = session.mul(x, &inv)?;

    let weight = broadcast_last_vector(session, weight, &target)?;
    if centered {
        let one = scalar_like(session, x, 1.0)?;
        let scale = session.add(&weight, &one)?;
        session.mul(&normed, &scale)
    } else {
        session.mul(&normed, &weight)
    }
}
