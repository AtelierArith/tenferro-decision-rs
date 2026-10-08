//! Tenferro-backed chunked Gated DeltaNet formulation (one value head).
//!
//! Follows `12_TENFERRO_GATED_DELTA.md` §7 and the Julia reference
//! `delta_attention` chunk loop. All heavy operations run through the eager
//! session (`dot_general` via `matmul`, `triangular_solve`, `exp`, reductions,
//! `rms_norm`, `silu`).

use tenferro_ad::{EagerSession, EagerTensor, Result, SliceConfig, Tensor};
use tenferro_linalg::EagerSessionLinalgExt;

use crate::reference::DeltaScanInputs;

/// Prepared tensors for one value-head chunk. All tensors belong to the same
/// eager runtime. Q/K are already L2-normalized and Q is scaled by 1/sqrt(key_dim).
/// Decay factors may be prepared entirely on device by the CUDA chunk kernel.
pub struct ChunkStepInputs<'a> {
    /// Persistent state, shape (value_dim, key_dim).
    pub state: &'a EagerTensor,
    /// Query/key matrices, shape (key_dim, chunk_length).
    pub q: &'a EagerTensor,
    pub k: &'a EagerTensor,
    /// Value and output-gate matrices, shape (value_dim, chunk_length).
    pub v: &'a EagerTensor,
    pub z: &'a EagerTensor,
    /// Beta and exp(cumulative log decay), shape (chunk_length,).
    pub beta: &'a EagerTensor,
    pub exp_decay: &'a EagerTensor,
    /// Lower-triangular exp(prefix[i]-prefix[j]), shape (chunk_length, chunk_length).
    pub pair_decay: &'a EagerTensor,
    /// exp(last prefix-prefix[j]), shape (chunk_length,).
    pub tail: &'a EagerTensor,
    /// Scalar exp(last prefix), shape ().
    pub final_decay: &'a EagerTensor,
    /// RMSNorm scale, shape (value_dim,).
    pub norm_weight: &'a EagerTensor,
    /// Prepared scalar normalization epsilon, inverse value width, and one.
    pub eps: &'a EagerTensor,
    pub inverse_value_dim: &'a EagerTensor,
    pub one: &'a EagerTensor,
}

/// Run the native contraction/solve/state-update/epilogue for one chunk.
/// Returns (next state, gated output), with shapes (value_dim, key_dim) and
/// (value_dim, chunk_length). All computation uses eager session operations;
/// this function creates no host constants, reads no tensor values onto the
/// host, and performs no explicit synchronization or backend fallback.
/// Shape/dtype/backend failures propagate as typed tenferro errors.
pub fn delta_scan_chunk_step(
    session: &mut EagerSession<'_>,
    inputs: ChunkStepInputs<'_>,
) -> Result<(EagerTensor, EagerTensor)> {
    let invalid = || {
        tenferro_ad::Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
            "delta_scan_chunk_step",
            "shape",
            "prepared chunk tensor shapes must agree and have nonzero dimensions",
        ))
    };
    let [kd, n] = inputs.q.shape() else {
        return Err(invalid());
    };
    let [vd, state_kd] = inputs.state.shape() else {
        return Err(invalid());
    };
    if *kd == 0 || *n == 0 || *vd == 0 || kd != state_kd {
        return Err(invalid());
    }
    for (tensor, shape) in [
        (inputs.k, vec![*kd, *n]),
        (inputs.v, vec![*vd, *n]),
        (inputs.z, vec![*vd, *n]),
        (inputs.beta, vec![*n]),
        (inputs.exp_decay, vec![*n]),
        (inputs.pair_decay, vec![*n, *n]),
        (inputs.tail, vec![*n]),
        (inputs.final_decay, vec![]),
        (inputs.norm_weight, vec![*vd]),
        (inputs.eps, vec![]),
        (inputs.inverse_value_dim, vec![]),
        (inputs.one, vec![]),
    ] {
        if tensor.shape() != shape {
            return Err(invalid());
        }
    }
    let pair_t = session.transpose(inputs.pair_decay, &[1, 0])?;
    let k_beta = session.mul(inputs.k, inputs.beta)?;
    let k_beta_t = session.transpose(&k_beta, &[1, 0])?;
    let system = session.matmul(&k_beta_t, inputs.k)?;
    let system = session.mul(&system, inputs.pair_decay)?;

    let v_beta = session.mul(inputs.v, inputs.beta)?;
    let rhs_values = session.transpose(&v_beta, &[1, 0])?;
    let new_values = session.triangular_solve(&system, &rhs_values, true, true, false, true)?;
    let k_beta_e = session.mul(&k_beta, inputs.exp_decay)?;
    let rhs_keys = session.transpose(&k_beta_e, &[1, 0])?;
    let reading_keys = session.triangular_solve(&system, &rhs_keys, true, true, false, true)?;

    let new_values_t = session.transpose(&new_values, &[1, 0])?;
    let reading_keys_t = session.transpose(&reading_keys, &[1, 0])?;
    let state_reading = session.matmul(inputs.state, &reading_keys_t)?;
    let corrections = session.sub(&new_values_t, &state_reading)?;
    let k_t = session.transpose(inputs.k, &[1, 0])?;
    let kq = session.matmul(&k_t, inputs.q)?;
    let intra = session.mul(&kq, &pair_t)?;
    let q_e = session.mul(inputs.q, inputs.exp_decay)?;
    let state_q = session.matmul(inputs.state, &q_e)?;
    let correction_intra = session.matmul(&corrections, &intra)?;
    let result = session.add(&state_q, &correction_intra)?;

    let ending_keys = session.mul(inputs.k, inputs.tail)?;
    let ending_keys_t = session.transpose(&ending_keys, &[1, 0])?;
    let state_scaled = session.mul(inputs.state, inputs.final_decay)?;
    let correction_keys = session.matmul(&corrections, &ending_keys_t)?;
    let next_state = session.add(&state_scaled, &correction_keys)?;

    let square_sum = session.reduce_sum_squares(&result, &[0])?;
    let mean_square = session.mul(&square_sum, inputs.inverse_value_dim)?;
    let denominator = session.add(&mean_square, inputs.eps)?;
    let inverse = session.rsqrt(&denominator)?;
    let normalized = session.mul(&result, &inverse)?;
    let weight = session.reshape(inputs.norm_weight, vec![inputs.norm_weight.shape()[0], 1])?;
    let normalized = session.mul(&normalized, &weight)?;
    let negative_z = session.neg(inputs.z)?;
    let exp_z = session.exp(&negative_z)?;
    let gate_denominator = session.add(&exp_z, inputs.one)?;
    let gate = session.div(inputs.one, &gate_denominator)?;
    let gated_z = session.mul(inputs.z, &gate)?;
    let output = session.mul(&normalized, &gated_z)?;
    Ok((next_state, output))
}

/// Run the chunked scan on `session` and return the head output as a
/// row-major `(value_dim, length)` vector.
pub fn delta_scan_chunked(
    session: &mut EagerSession<'_>,
    inputs: &DeltaScanInputs<'_>,
    chunk_size: usize,
) -> Result<Vec<f32>> {
    let kd = inputs.key_dim;
    let vd = inputs.value_dim;
    let length = inputs.length;

    let k = constant(session, &[kd, length], &col_major(kd, length, inputs.k)?)?;
    let q = constant(session, &[kd, length], &col_major(kd, length, inputs.q)?)?;
    let v = constant(session, &[vd, length], &col_major(vd, length, inputs.v)?)?;
    let z = constant(session, &[vd, length], &col_major(vd, length, inputs.z)?)?;
    let beta = constant(session, &[length], inputs.beta)?;
    let norm_weight = constant(session, &[vd], inputs.norm)?;
    let eps = constant(session, &[], &[inputs.eps])?;
    let inverse_value_dim = constant(session, &[], &[1.0 / vd as f32])?;
    let one = constant(session, &[], &[1.0])?;

    let mut state = constant(session, &[vd, kd], &vec![0.0f32; vd * kd])?;
    let mut out = vec![0.0f32; vd * length];

    let mut start = 0usize;
    while start < length {
        let end = (start + chunk_size).min(length);
        let n = end - start;

        let qc = slice_cols(session, &q, kd, start, end)?;
        let kc = slice_cols(session, &k, kd, start, end)?;
        let vc = slice_cols(session, &v, vd, start, end)?;
        let zc = slice_cols(session, &z, vd, start, end)?;
        let bc = slice_range(session, &beta, start, end)?;

        // Cumulative decay (host) and the pair-decay matrix.
        let mut cumulative = vec![0.0f32; n];
        let mut acc = 0.0f32;
        for (index, value) in cumulative.iter_mut().enumerate() {
            acc += inputs.decay[start + index];
            *value = acc;
        }
        let cumulative = constant(session, &[n], &cumulative)?;
        let exp_decay = session.exp(&cumulative)?;

        let mut pair_decay = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                if i >= j {
                    pair_decay[i + j * n] = (cumulative_host(inputs, start, i)
                        - cumulative_host(inputs, start, j))
                    .exp();
                }
            }
        }
        let pair_decay = constant(session, &[n, n], &pair_decay)?;
        // Compute tail weights from log-prefix differences, never log(exp(sum)):
        // the latter loses finite tails when the final factor underflows.
        let cumulative_last = inputs.decay[start..end].iter().sum::<f32>();
        let tail: Vec<f32> = (0..n)
            .map(|j| (cumulative_last - cumulative_host(inputs, start, j)).exp())
            .collect();
        let tail = constant(session, &[n], &tail)?;
        let final_decay = constant(session, &[], &[cumulative_last.exp()])?;
        let (next_state, chunk_out) = delta_scan_chunk_step(
            session,
            ChunkStepInputs {
                state: &state,
                q: &qc,
                k: &kc,
                v: &vc,
                z: &zc,
                beta: &bc,
                exp_decay: &exp_decay,
                pair_decay: &pair_decay,
                tail: &tail,
                final_decay: &final_decay,
                norm_weight: &norm_weight,
                eps: &eps,
                inverse_value_dim: &inverse_value_dim,
                one: &one,
            },
        )?;
        state = next_state;

        let host = session.duplicate_value(&chunk_out)?;
        let values = host.as_slice::<f32>()?;
        for column in 0..n {
            for row in 0..vd {
                out[row * length + start + column] = values[row + column * vd];
            }
        }

        start = end;
    }
    Ok(out)
}

fn cumulative_host(inputs: &DeltaScanInputs<'_>, start: usize, index: usize) -> f32 {
    inputs.decay[start..=start + index].iter().sum()
}

pub(crate) fn constant(
    session: &mut EagerSession<'_>,
    shape: &[usize],
    data: &[f32],
) -> Result<EagerTensor> {
    session.constant_from(Tensor::from_vec_col_major(shape.to_vec(), data.to_vec())?)
}

pub(crate) fn col_major(rows: usize, cols: usize, row_major: &[f32]) -> Result<Vec<f32>> {
    if row_major.len() != rows * cols {
        return Err(tenferro_ad::Error::TensorRuntime(
            tenferro_tensor::Error::invalid_argument(
                "tenferro-gated-delta",
                "input",
                "row-major input has the wrong length",
            ),
        ));
    }
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            out[row + col * rows] = row_major[row * cols + col];
        }
    }
    Ok(out)
}

fn slice_cols(
    session: &mut EagerSession<'_>,
    input: &EagerTensor,
    rows: usize,
    start: usize,
    end: usize,
) -> Result<EagerTensor> {
    session.slice(
        input,
        SliceConfig {
            starts: vec![0, start],
            limits: vec![rows, end],
            strides: vec![1, 1],
        },
    )
}

fn slice_range(
    session: &mut EagerSession<'_>,
    input: &EagerTensor,
    start: usize,
    end: usize,
) -> Result<EagerTensor> {
    session.slice(
        input,
        SliceConfig {
            starts: vec![start],
            limits: vec![end],
            strides: vec![1],
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_step_rejects_scalar_norm_weight_with_typed_error() {
        let runtime =
            tenferro_ad::EagerRuntime::with_cpu_backend(tenferro_cpu::CpuBackend::new()).unwrap();
        runtime
            .with_eager_session(|session| {
                let state = constant(session, &[2, 3], &[0.0; 6])?;
                let q = constant(session, &[3, 4], &[0.0; 12])?;
                let v = constant(session, &[2, 4], &[0.0; 8])?;
                let beta = constant(session, &[4], &[0.0; 4])?;
                let pair = constant(session, &[4, 4], &[0.0; 16])?;
                let scalar = constant(session, &[], &[1.0])?;
                let error = delta_scan_chunk_step(
                    session,
                    ChunkStepInputs {
                        state: &state,
                        q: &q,
                        k: &q,
                        v: &v,
                        z: &v,
                        beta: &beta,
                        exp_decay: &beta,
                        pair_decay: &pair,
                        tail: &beta,
                        final_decay: &scalar,
                        norm_weight: &scalar,
                        eps: &scalar,
                        inverse_value_dim: &scalar,
                        one: &scalar,
                    },
                )
                .expect_err("scalar norm weight must be rejected");
                assert!(matches!(error, tenferro_ad::Error::TensorRuntime(_)));
                Ok::<(), tenferro_ad::Error>(())
            })
            .unwrap()
            .unwrap();
    }
}
