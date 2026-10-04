//! Reference multi-head attention.
//!
//! Layout is `(B, H, L, head_dim)`; the result has the same layout. Scores use
//! `dot_general` batch dimensions and produce the batch-trailing
//! `(Lq, Lk, B, H)` convention, so the softmax axis is `Lk` (axis 1).

use tenferro_ad::{DotGeneralConfig, EagerSession, EagerTensor, Result};

use crate::softmax::{masked_softmax, softmax};

/// Scaled dot-product attention.
///
/// `mask`, when given, has shape `(Lq, Lk)` and selects keys to keep
/// (`true`) or mask out (`false`). The default scale is `1 / sqrt(head_dim)`.
pub fn attention(
    session: &mut EagerSession<'_>,
    q: &EagerTensor,
    k: &EagerTensor,
    v: &EagerTensor,
    mask: Option<&EagerTensor>,
    scale: Option<f64>,
) -> Result<EagerTensor> {
    let head_dim = q.shape()[3] as f64;
    let scale = scale.unwrap_or(1.0 / head_dim.sqrt());

    let scores = session.dot_general(
        q,
        k,
        DotGeneralConfig {
            lhs_contracting_dims: [3].as_slice().into(),
            rhs_contracting_dims: [3].as_slice().into(),
            lhs_batch_dims: [0, 1].as_slice().into(),
            rhs_batch_dims: [0, 1].as_slice().into(),
        },
    )?;
    let scores = session.scale_real(&scores, scale)?;

    let probs = match mask {
        Some(mask) => {
            let target = scores.shape().to_vec();
            let dims: Vec<usize> = (0..mask.shape().len()).collect();
            let mask = session.broadcast_in_dim(mask, &target, &dims)?;
            masked_softmax(session, &scores, &mask, 1)?
        }
        None => softmax(session, &scores, 1)?,
    };

    // probs: (Lq, Lk, B, H), v: (B, H, Lk, hd) -> (Lq, hd, B, H)
    let out = session.dot_general(
        &probs,
        v,
        DotGeneralConfig {
            lhs_contracting_dims: [1].as_slice().into(),
            rhs_contracting_dims: [2].as_slice().into(),
            lhs_batch_dims: [2, 3].as_slice().into(),
            rhs_batch_dims: [0, 1].as_slice().into(),
        },
    )?;
    // -> (B, H, Lq, hd)
    session.transpose(&out, &[2, 3, 0, 1])
}
