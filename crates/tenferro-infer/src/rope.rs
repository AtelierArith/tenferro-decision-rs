//! Rotary position embeddings.
//!
//! Both variants operate on `x` whose last axis is the head dimension and whose
//! second-to-last axis is the sequence. Rotary pairs are `(i, i + width/2)` and
//! positions are `0 .. L-1`.
//!
//! - [`rope_modernbert`] rotates the whole head dimension with a per-layer-kind
//!   base (Laya: 160000 full attention / 10000 sliding).
//! - [`rope_qwen_partial`] rotates only the first `rotary_dim` channels and
//!   leaves the tail unchanged (Jeff partial RoPE).

use tenferro_ad::{DType, EagerSession, EagerTensor, Error as AdError, Result, Tensor};

use crate::util::{slice_axis, unsupported};

fn table(dtype: DType, data: &[f64], shape: Vec<usize>) -> Result<Tensor> {
    match dtype {
        DType::F64 => Ok(Tensor::from_vec_col_major(shape, data.to_vec())?),
        DType::F32 => Ok(Tensor::from_vec_col_major(
            shape,
            data.iter().map(|value| *value as f32).collect(),
        )?),
        other => Err(unsupported(other, "floating point")),
    }
}

fn invalid(message: impl Into<String>) -> AdError {
    AdError::TensorRuntime(tenferro_tensor::Error::invalid_argument(
        "tenferro-infer::rope",
        "input",
        message,
    ))
}

/// Rotate the whole head dimension of `x` with pairs `(i, i + head/2)`.
pub fn rope_modernbert(
    session: &mut EagerSession<'_>,
    x: &EagerTensor,
    base: f64,
) -> Result<EagerTensor> {
    rotate_pair_half(session, x, base, x.shape()[x.shape().len() - 1])
}

/// Rotate the first `rotary_dim` channels of `x`, leaving the tail unchanged.
pub fn rope_qwen_partial(
    session: &mut EagerSession<'_>,
    x: &EagerTensor,
    base: f64,
    rotary_dim: usize,
) -> Result<EagerTensor> {
    let rank = x.shape().len();
    let head = x.shape()[rank - 1];
    if rotary_dim > head {
        return Err(invalid(format!(
            "rotary_dim {rotary_dim} exceeds head dimension {head}"
        )));
    }
    if rotary_dim == head {
        return rope_modernbert(session, x, base);
    }

    let rotated = slice_axis(session, x, rank - 1, 0, rotary_dim)?;
    let rotated = rotate_pair_half(session, &rotated, base, rotary_dim)?;
    let tail = slice_axis(session, x, rank - 1, rotary_dim, head - rotary_dim)?;
    session.concatenate(&[&rotated, &tail], rank - 1)
}

fn rotate_pair_half(
    session: &mut EagerSession<'_>,
    x: &EagerTensor,
    base: f64,
    width: usize,
) -> Result<EagerTensor> {
    if width == 0 || width % 2 != 0 {
        return Err(invalid(format!(
            "rotary width must be a positive even number, got {width}"
        )));
    }
    let rank = x.shape().len();
    let seq = x.shape()[rank - 2];
    let half = width / 2;

    // Host-side cos/sin tables of shape (seq, half), column-major.
    let mut cos = vec![0.0f64; seq * half];
    let mut sin = vec![0.0f64; seq * half];
    for position in 0..seq {
        for i in 0..half {
            let theta = (position as f64) * base.powf(-2.0 * (i as f64) / (width as f64));
            cos[i * seq + position] = theta.cos();
            sin[i * seq + position] = theta.sin();
        }
    }

    let first = slice_axis(session, x, rank - 1, 0, half)?;
    let second = slice_axis(session, x, rank - 1, half, half)?;
    let target = first.shape().to_vec();
    let dims = [rank - 2, rank - 1];

    let cos = session.constant_from_host(table(x.dtype(), &cos, vec![seq, half])?)?;
    let sin = session.constant_from_host(table(x.dtype(), &sin, vec![seq, half])?)?;
    let cos = session.broadcast_in_dim(&cos, &target, &dims)?;
    let sin = session.broadcast_in_dim(&sin, &target, &dims)?;

    let a = session.mul(&first, &cos)?;
    let b = session.mul(&second, &sin)?;
    let out1 = session.sub(&a, &b)?;

    let c = session.mul(&first, &sin)?;
    let d = session.mul(&second, &cos)?;
    let out2 = session.add(&c, &d)?;

    session.concatenate(&[&out1, &out2], rank - 1)
}
