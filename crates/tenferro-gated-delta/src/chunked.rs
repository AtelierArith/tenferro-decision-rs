//! Tenferro-backed chunked Gated DeltaNet formulation (one value head).
//!
//! Follows `12_TENFERRO_GATED_DELTA.md` §7 and the Julia reference
//! `delta_attention` chunk loop. All heavy operations run through the eager
//! session (`dot_general` via `matmul`, `triangular_solve`, `exp`, reductions,
//! `rms_norm`, `silu`).

use tenferro_ad::{EagerSession, EagerTensor, Result, SliceConfig, Tensor};
use tenferro_infer::{activation, norm};
use tenferro_linalg::EagerSessionLinalgExt;

use crate::reference::DeltaScanInputs;

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
        let pair_decay_t = session.transpose(&pair_decay, &[1, 0])?;

        // system = (kc * beta)^T @ kc .* pair_decay
        let kc_beta = session.mul(&kc, &bc)?;
        let kc_beta_t = session.transpose(&kc_beta, &[1, 0])?;
        let system = session.matmul(&kc_beta_t, &kc)?;
        let system = session.mul(&system, &pair_decay)?;

        // rhs_values = (vc * beta)^T ; rhs_keys = (kc * beta * exp_decay)^T
        let vc_beta = session.mul(&vc, &bc)?;
        let rhs_values = session.transpose(&vc_beta, &[1, 0])?;
        let new_values = session.triangular_solve(&system, &rhs_values, true, true, false, true)?;

        let kc_beta_e = session.mul(&kc_beta, &exp_decay)?;
        let rhs_keys = session.transpose(&kc_beta_e, &[1, 0])?;
        let reading_keys = session.triangular_solve(&system, &rhs_keys, true, true, false, true)?;

        // corrections = new_values^T - state @ reading_keys^T
        let new_values_t = session.transpose(&new_values, &[1, 0])?;
        let reading_keys_t = session.transpose(&reading_keys, &[1, 0])?;
        let state_reading = session.matmul(&state, &reading_keys_t)?;
        let corrections = session.sub(&new_values_t, &state_reading)?;

        // intra = (kc^T @ qc) .* pair_decay^T
        let kc_t = session.transpose(&kc, &[1, 0])?;
        let kq = session.matmul(&kc_t, &qc)?;
        let intra = session.mul(&kq, &pair_decay_t)?;

        // result = state @ (qc .* exp_decay) + corrections @ intra
        let qc_e = session.mul(&qc, &exp_decay)?;
        let state_q = session.matmul(&state, &qc_e)?;
        let correction_intra = session.matmul(&corrections, &intra)?;
        let result = session.add(&state_q, &correction_intra)?;

        // state update.
        //
        // `tail[j] = exp(cumsum_last - cumsum_j)` from the host cumulative
        // values. Do **not** route this through `exp(sum).ln()`: a strongly
        // negative decay sum (e.g. ~-600 over a 64-token chunk) underflows the
        // chunk decay to zero, and `ln(0) = -inf` would zero the state update.
        let cumulative_last = inputs.decay[start..end].iter().sum::<f32>();
        let final_decay = cumulative_last.exp();
        let tail: Vec<f32> = (0..n)
            .map(|j| (cumulative_last - cumulative_host(inputs, start, j)).exp())
            .collect();
        let tail = constant(session, &[n], &tail)?;
        let ending_keys = session.mul(&kc, &tail)?;
        let ending_keys_t = session.transpose(&ending_keys, &[1, 0])?;
        let state_scaled = session.scale_real(&state, final_decay as f64)?;
        let correction_keys = session.matmul(&corrections, &ending_keys_t)?;
        state = session.add(&state_scaled, &correction_keys)?;

        // Non-centered RMSNorm over the value dimension, then the output gate.
        let result_t = session.transpose(&result, &[1, 0])?;
        let normalized_t =
            norm::rms_norm(session, &result_t, &norm_weight, false, inputs.eps as f64)?;
        let normalized = session.transpose(&normalized_t, &[1, 0])?;
        let gated_z = activation::silu(session, &zc)?;
        let chunk_out = session.mul(&normalized, &gated_z)?;

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
