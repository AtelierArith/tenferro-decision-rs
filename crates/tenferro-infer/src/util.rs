use tenferro_ad::{
    DType, EagerSession, EagerTensor, Error as AdError, Result, SliceConfig, Tensor,
};

/// Build a rank-0 constant tensor of the same floating dtype as `like`.
pub(crate) fn scalar_like(
    session: &mut EagerSession<'_>,
    like: &EagerTensor,
    value: f64,
) -> Result<EagerTensor> {
    let tensor = match like.dtype() {
        DType::F64 => Tensor::from_vec_col_major(vec![], vec![value])?,
        DType::F32 => Tensor::from_vec_col_major(vec![], vec![value as f32])?,
        dtype => return Err(unsupported(dtype, "floating point")),
    };
    session.constant_from_host(tensor)
}

/// Insert a singleton at `axis` and broadcast back to `target`.
pub(crate) fn broadcast_axis(
    session: &mut EagerSession<'_>,
    input: &EagerTensor,
    target: &[usize],
    axis: usize,
) -> Result<EagerTensor> {
    let rank = target.len();
    let mut shape = input.shape().to_vec();
    shape.insert(axis, 1);
    let reshaped = session.reshape(input, shape)?;
    let dims: Vec<usize> = (0..rank).collect();
    session.broadcast_in_dim(&reshaped, target, &dims)
}

/// Broadcast a rank-1 vector to the last axis of `target` (norm weights/biases).
pub(crate) fn broadcast_last_vector(
    session: &mut EagerSession<'_>,
    vector: &EagerTensor,
    target: &[usize],
) -> Result<EagerTensor> {
    let rank = target.len();
    let mut shape = vec![1usize; rank];
    shape[rank - 1] = vector.shape()[0];
    let reshaped = session.reshape(vector, shape)?;
    let dims: Vec<usize> = (0..rank).collect();
    session.broadcast_in_dim(&reshaped, target, &dims)
}

/// Slice `len` elements from `start` along `axis`, full range elsewhere.
pub(crate) fn slice_axis(
    session: &mut EagerSession<'_>,
    input: &EagerTensor,
    axis: usize,
    start: usize,
    len: usize,
) -> Result<EagerTensor> {
    let mut starts = vec![0usize; input.shape().len()];
    let mut limits = input.shape().to_vec();
    let strides = vec![1usize; input.shape().len()];
    starts[axis] = start;
    limits[axis] = start + len;
    session.slice(
        input,
        SliceConfig {
            starts,
            limits,
            strides,
        },
    )
}

/// A large negative constant of the same dtype as `like`, used to mask scores.
pub(crate) fn neg_inf_like(
    session: &mut EagerSession<'_>,
    like: &EagerTensor,
) -> Result<EagerTensor> {
    scalar_like(session, like, -1.0e30)
}

pub(crate) fn unsupported(dtype: DType, expected: &str) -> AdError {
    AdError::TensorRuntime(tenferro_tensor::Error::unsupported(
        "tenferro-infer",
        format!("unsupported dtype {dtype:?}; expected {expected}"),
    ))
}
