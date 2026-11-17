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
    let scores = crate::util::scale_real(session, &scores, scale)?;

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

/// Scaled dot-product attention with an **additive** score bias.
///
/// `bias` is a floating tensor of the scores' dtype with shape `(Lq, Lk)` (or
/// any leading prefix of the batch-trailing score shape, like [`attention`]'s
/// mask), added to the scaled scores before the softmax: `0` keeps a key and a
/// large negative value (e.g. `-1e30`) masks it. Unlike a Bool `where_select`
/// mask this needs no Bool broadcast/materialization, which backends such as
/// the pinned CUDA provider do not support. A row whose keys are all masked
/// yields a uniform distribution, as with [`attention`]'s mask.
pub fn attention_with_bias(
    session: &mut EagerSession<'_>,
    q: &EagerTensor,
    k: &EagerTensor,
    v: &EagerTensor,
    bias: &EagerTensor,
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
    let scores = crate::util::scale_real(session, &scores, scale)?;
    let target = scores.shape().to_vec();
    let dims: Vec<usize> = (0..bias.shape().len()).collect();
    let bias = session.broadcast_in_dim(bias, &target, &dims)?;
    let scores = session.add(&scores, &bias)?;
    let probs = softmax(session, &scores, 1)?;

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
    session.transpose(&out, &[2, 3, 0, 1])
}

/// Upload a host keep-mask as an additive F32 attention bias: `0` where
/// `keep` is true and `-1e30` elsewhere, in the given column-major shape.
pub fn keep_bias_f32(
    session: &mut EagerSession<'_>,
    shape: Vec<usize>,
    keep: &[bool],
) -> Result<EagerTensor> {
    let values: Vec<f32> = keep
        .iter()
        .map(|keep| if *keep { 0.0 } else { -1.0e30 })
        .collect();
    session.constant_from_host(tenferro_ad::Tensor::from_vec_col_major(shape, values)?)
}
