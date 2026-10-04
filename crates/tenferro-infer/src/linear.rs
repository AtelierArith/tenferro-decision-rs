//! Prepared linear (dense) layers.

use tenferro_ad::{DotGeneralConfig, EagerSession, EagerTensor, Result};

/// `x @ weight`, contracting the last axis of `x` with the first axis of
/// `weight`.
///
/// `weight` has shape `(in_features, out_features)` (the engine's canonical
/// `(in, out)` layout). The output keeps `x`'s leading axes and appends
/// `out_features`.
pub fn linear(
    session: &mut EagerSession<'_>,
    x: &EagerTensor,
    weight: &EagerTensor,
) -> Result<EagerTensor> {
    let rank = x.shape().len();
    let config = DotGeneralConfig {
        lhs_contracting_dims: [rank - 1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    };
    session.dot_general(x, weight, config)
}
