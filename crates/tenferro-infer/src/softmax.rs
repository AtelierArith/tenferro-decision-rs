//! Stable softmax and masked softmax over an arbitrary axis.

use tenferro_ad::{EagerSession, EagerTensor, Result};

use crate::util::{broadcast_axis, neg_inf_like};

/// Numerically stable softmax over `axis`.
pub fn softmax(
    session: &mut EagerSession<'_>,
    x: &EagerTensor,
    axis: usize,
) -> Result<EagerTensor> {
    let target = x.shape().to_vec();
    let max = session.reduce_max(x, Some(&[axis]))?;
    let max = broadcast_axis(session, &max, &target, axis)?;
    let shifted = session.sub(x, &max)?;
    let exp = session.exp(&shifted)?;
    let sum = session.reduce_sum(&exp, Some(&[axis]))?;
    let sum = broadcast_axis(session, &sum, &target, axis)?;
    session.div(&exp, &sum)
}

/// Softmax that treats positions where `keep` is `false` as `-inf`.
///
/// `keep` must be broadcast-compatible with `x` (typically the same shape).
pub fn masked_softmax(
    session: &mut EagerSession<'_>,
    x: &EagerTensor,
    keep: &EagerTensor,
    axis: usize,
) -> Result<EagerTensor> {
    let neg = neg_inf_like(session, x)?;
    let masked = session.where_select(keep, x, &neg)?;
    softmax(session, &masked, axis)
}
