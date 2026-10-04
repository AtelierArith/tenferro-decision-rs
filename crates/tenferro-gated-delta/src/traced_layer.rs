//! Graph-compiled (traced) Gated DeltaNet layer.
//!
//! Same math as [`crate::tensor_layer`], expressed on [`TracedTensor`] so a
//! `GraphCompiler` can lower it to an `ExecProgram`, fuse consecutive
//! elementwise ops into SIMD kernels, and run it through a `Runtime`. The eager
//! path stays the reference; this module exists to measure the fused build.

use tenferro_linalg::TracedTensorLinalgExt;
use tenferro_runtime::{
    DType, DotGeneralConfig, PadConfig, Result as R, SliceConfig, TracedTensor,
};

use crate::config::GatedDeltaConfig;
use crate::layer::GatedDeltaWeights;

/// Traced weight tensors for one Gated DeltaNet layer.
#[derive(Clone)]
pub struct GatedDeltaTracedWeights {
    pub qkv: TracedTensor,
    pub z: TracedTensor,
    pub a: TracedTensor,
    pub b: TracedTensor,
    pub conv: TracedTensor,
    pub a_decay: TracedTensor,
    pub dt_bias: TracedTensor,
    pub norm: TracedTensor,
    pub out_proj: TracedTensor,
}

/// Column-major `[rows, cols]` traced constant from row-major `data`.
fn col(shape: [usize; 2], row_major: &[f32]) -> R<TracedTensor> {
    let [rows, cols] = shape;
    let mut column_major = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for column in 0..cols {
            column_major[row + column * rows] = row_major[row * cols + column];
        }
    }
    TracedTensor::from_vec_col_major(vec![rows, cols], column_major)
}

/// Column-major traced constant from data that is already column-major.
fn col_major(shape: Vec<usize>, data: &[f32]) -> R<TracedTensor> {
    TracedTensor::from_vec_col_major(shape, data.to_vec())
}

/// Build the traced weights for a layer from the host weights.
pub fn prepare_traced_weights(
    cfg: &GatedDeltaConfig,
    weights: &GatedDeltaWeights,
) -> R<GatedDeltaTracedWeights> {
    let key_width = cfg.key_dim * cfg.key_heads;
    let value_width = cfg.value_dim * cfg.value_heads;
    let conv_channels = 2 * key_width + value_width;
    Ok(GatedDeltaTracedWeights {
        qkv: col([cfg.hidden, conv_channels], &weights.qkv)?,
        z: col([cfg.hidden, value_width], &weights.z)?,
        a: col([cfg.hidden, cfg.value_heads], &weights.a)?,
        b: col([cfg.hidden, cfg.value_heads], &weights.b)?,
        conv: col_major(vec![conv_channels, cfg.conv_taps], &weights.conv)?,
        a_decay: col_major(vec![cfg.value_heads], &weights.a_decay)?,
        dt_bias: col_major(vec![cfg.value_heads], &weights.dt_bias)?,
        norm: col_major(vec![cfg.value_dim], &weights.norm)?,
        out_proj: col([value_width, cfg.hidden], &weights.out_proj)?,
    })
}

fn scalar(value: f32) -> R<TracedTensor> {
    TracedTensor::from_vec_col_major(vec![1], vec![value])
}

/// `y = weightᵀ x` for `x (in, L)` and prepared `weight (in, out)`.
fn linear_col(x: &TracedTensor, weight: &TracedTensor) -> R<TracedTensor> {
    let y = x.dot_general(
        weight,
        DotGeneralConfig {
            lhs_contracting_dims: [0].as_slice().into(),
            rhs_contracting_dims: [0].as_slice().into(),
            lhs_batch_dims: [].as_slice().into(),
            rhs_batch_dims: [].as_slice().into(),
        },
    )?;
    y.transpose(&[1, 0])
}

/// Batched `a (H, m, k) @ b (H, k, n) -> (H, m, n)` (tenferro emits batch-trailing).
fn bmm(a: &TracedTensor, b: &TracedTensor) -> R<TracedTensor> {
    let out = a.dot_general(
        b,
        DotGeneralConfig {
            lhs_contracting_dims: [2].as_slice().into(),
            rhs_contracting_dims: [1].as_slice().into(),
            lhs_batch_dims: [0].as_slice().into(),
            rhs_batch_dims: [0].as_slice().into(),
        },
    )?;
    out.transpose(&[2, 0, 1])
}

fn slice_2d(input: &TracedTensor, row_start: usize, rows: usize, cols: usize) -> R<TracedTensor> {
    input.slice(SliceConfig {
        starts: vec![row_start, 0],
        limits: vec![row_start + rows, cols],
        strides: vec![1, 1],
    })
}

fn slice_time_rows(input: &TracedTensor, rows: usize, start: usize, end: usize) -> R<TracedTensor> {
    input.slice(SliceConfig {
        starts: vec![0, start],
        limits: vec![rows, end],
        strides: vec![1, 1],
    })
}

fn slice_time(
    input: &TracedTensor,
    outer: usize,
    rows: usize,
    start: usize,
    end: usize,
) -> R<TracedTensor> {
    input.slice(SliceConfig {
        starts: vec![0, 0, start],
        limits: vec![outer, rows, end],
        strides: vec![1, 1, 1],
    })
}

fn slice_features(
    input: &TracedTensor,
    heads: usize,
    n: usize,
    start: usize,
    end: usize,
) -> R<TracedTensor> {
    input.slice(SliceConfig {
        starts: vec![0, 0, start],
        limits: vec![heads, n, end],
        strides: vec![1, 1, 1],
    })
}

fn softplus(x: &TracedTensor) -> R<TracedTensor> {
    let zero = scalar(0.0)?;
    let one = scalar(1.0)?;
    let positive = x.maximum(&zero)?;
    let abs = x.abs()?;
    let neg_abs = abs.scale_real(-1.0)?;
    let exp = neg_abs.exp()?;
    let shifted = exp.add(&one)?;
    let log = shifted.log()?;
    positive.add(&log)
}

fn causal_depthwise_silu(
    x: &TracedTensor,
    conv: &TracedTensor,
    channels: usize,
    length: usize,
    taps: usize,
) -> R<TracedTensor> {
    let mut acc: Option<TracedTensor> = None;
    for tap in 0..taps {
        let lag = taps - 1 - tap;
        let shifted = if lag == 0 {
            x.clone()
        } else {
            let padded = x.pad(PadConfig {
                edge_padding_low: vec![0, lag as i64],
                edge_padding_high: vec![0, 0],
                interior_padding: vec![0, 0],
            })?;
            padded.slice(SliceConfig {
                starts: vec![0, 0],
                limits: vec![channels, length],
                strides: vec![1, 1],
            })?
        };
        let weight = conv.slice(SliceConfig {
            starts: vec![0, tap],
            limits: vec![channels, tap + 1],
            strides: vec![1, 1],
        })?;
        let term = shifted.mul(&weight)?;
        acc = Some(match acc {
            None => term,
            Some(previous) => previous.add(&term)?,
        });
    }
    let sum = acc.expect("conv_taps >= 1");
    // silu(sum) = sum * sigmoid(sum) = sum / (1 + exp(-sum))
    let one = scalar(1.0)?;
    let neg = sum.neg()?;
    let exp = neg.exp()?;
    let denom = exp.add(&one)?;
    sum.div(&denom)
}

fn l2_normalize_heads(x: &TracedTensor, heads: usize, length: usize, eps: f64) -> R<TracedTensor> {
    let sum_sq = x.reduce_sum_squares(&[1])?; // (heads, length)
    let eps = TracedTensor::from_vec_col_major(vec![length], vec![eps as f32; length])?;
    let denom = sum_sq.add(&eps)?;
    let inv = denom.rsqrt()?;
    let inv = inv.reshape(vec![heads, 1, length])?;
    x.mul(&inv)
}

fn lower_tri_ones(n: usize) -> R<TracedTensor> {
    let ones = TracedTensor::from_vec_col_major(vec![n, n], vec![1.0f32; n * n])?;
    ones.tril(0)
}

fn slice_head(input: &TracedTensor, h: usize, rows: usize, cols: usize) -> R<TracedTensor> {
    let view = input.slice(SliceConfig {
        starts: vec![h, 0, 0],
        limits: vec![h + 1, rows, cols],
        strides: vec![1, 1, 1],
    })?;
    view.reshape(vec![rows, cols])
}

fn solve_unit_lower_heads(
    system: &TracedTensor,
    rhs: &TracedTensor,
    heads: usize,
    n: usize,
    cols: usize,
) -> R<TracedTensor> {
    let mut solved = Vec::with_capacity(heads);
    for h in 0..heads {
        let a = slice_head(system, h, n, n)?;
        let b = slice_head(rhs, h, n, cols)?;
        let x = a.triangular_solve(&b, true, true, false, true)?;
        solved.push(x.reshape(vec![1, n, cols])?);
    }
    if solved.len() == 1 {
        let single = solved.pop().expect("one head");
        return single.reshape(vec![1, n, cols]);
    }
    let refs: Vec<&TracedTensor> = solved.iter().collect();
    TracedTensor::concatenate(&refs, 0)
}

#[allow(clippy::too_many_arguments)]
fn delta_scan_chunked(
    key_dim: usize,
    value_dim: usize,
    length: usize,
    chunk_size: usize,
    heads: usize,
    q: &TracedTensor,
    k: &TracedTensor,
    v: &TracedTensor,
    z: &TracedTensor,
    beta: &TracedTensor,
    decay: &TracedTensor,
    norm_weight: &TracedTensor,
    eps: f64,
) -> R<TracedTensor> {
    let neg_big = scalar(-1.0e30)?;
    let zero = scalar(0.0)?;
    let mut state = TracedTensor::from_vec_col_major(
        vec![heads, value_dim, key_dim],
        vec![0.0f32; heads * value_dim * key_dim],
    )?;
    let mut chunks: Vec<TracedTensor> = Vec::new();

    let mut start = 0usize;
    while start < length {
        let end = (start + chunk_size).min(length);
        let n = end - start;

        let qc = slice_time(q, heads, key_dim, start, end)?;
        let kc = slice_time(k, heads, key_dim, start, end)?;
        let vc = slice_time(v, heads, value_dim, start, end)?;
        let zc = slice_time(z, heads, value_dim, start, end)?;
        let bc = slice_time_rows(beta, heads, start, end)?;
        let dc = slice_time_rows(decay, heads, start, end)?;

        let lower = lower_tri_ones(n)?;
        let lower = lower.broadcast_in_dim(&[heads, n, n], &[1, 2])?;
        let dc_col = dc.reshape(vec![heads, n, 1])?;
        let cumulative = bmm(&lower, &dc_col)?;
        let cumulative = cumulative.reshape(vec![heads, n])?;
        let exp_decay = cumulative.exp()?;

        let cum_col = cumulative.reshape(vec![heads, n, 1])?;
        let cum_row = cumulative.reshape(vec![heads, 1, n])?;
        let diff = cum_col.sub(&cum_row)?;
        let diff = diff.clamp(&neg_big, &zero)?;
        let pair = diff.exp()?;
        let pair_decay = pair.mul(&lower)?;
        let pair_decay_t = pair_decay.transpose(&[0, 2, 1])?;

        let bc_k = bc.reshape(vec![heads, 1, n])?;
        let kc_beta = kc.mul(&bc_k)?;
        let kc_beta_t = kc_beta.transpose(&[0, 2, 1])?;
        let system = bmm(&kc_beta_t, &kc)?;
        let system = system.mul(&pair_decay)?;

        let vc_beta = vc.mul(&bc_k)?;
        let rhs_values = vc_beta.transpose(&[0, 2, 1])?;
        let e_k = exp_decay.reshape(vec![heads, 1, n])?;
        let kc_beta_e = kc_beta.mul(&e_k)?;
        let rhs_keys = kc_beta_e.transpose(&[0, 2, 1])?;

        let rhs = TracedTensor::concatenate(&[&rhs_values, &rhs_keys], 2)?;
        let solved = solve_unit_lower_heads(&system, &rhs, heads, n, value_dim + key_dim)?;
        let new_values = slice_features(&solved, heads, n, 0, value_dim)?;
        let reading_keys = slice_features(&solved, heads, n, value_dim, value_dim + key_dim)?;

        let new_values_t = new_values.transpose(&[0, 2, 1])?;
        let reading_keys_t = reading_keys.transpose(&[0, 2, 1])?;
        let state_reading = bmm(&state, &reading_keys_t)?;
        let corrections = new_values_t.sub(&state_reading)?;

        let kc_t = kc.transpose(&[0, 2, 1])?;
        let kq = bmm(&kc_t, &qc)?;
        let intra = kq.mul(&pair_decay_t)?;

        let qc_e = qc.mul(&e_k)?;
        let state_q = bmm(&state, &qc_e)?;
        let correction_intra = bmm(&corrections, &intra)?;
        let result = state_q.add(&correction_intra)?;

        let cum_last = slice_time_rows(&cumulative, heads, n - 1, n)?;
        let tail = cum_last.sub(&cumulative)?;
        let tail = tail.exp()?;
        let tail_k = tail.reshape(vec![heads, 1, n])?;
        let final_decay = cum_last.exp()?;
        let final_decay = final_decay.reshape(vec![heads, 1, 1])?;
        let ending_keys = kc.mul(&tail_k)?;
        let ending_keys_t = ending_keys.transpose(&[0, 2, 1])?;
        let state_scaled = state.mul(&final_decay)?;
        let correction_keys = bmm(&corrections, &ending_keys_t)?;
        state = state_scaled.add(&correction_keys)?;

        let result_t = result.transpose(&[0, 2, 1])?; // (H, n, vd)
        let mean = result_t
            .mul(&result_t)?
            .reduce_sum(Some(&[2]))?
            .scale_real(1.0 / value_dim as f64)?
            .add(&scalar(eps as f32)?)?;
        let inv = mean.rsqrt()?;
        let inv = inv.reshape(vec![heads, n, 1])?;
        let normalized_t = result_t.mul(&inv)?.mul(norm_weight)?;
        let normalized = normalized_t.transpose(&[0, 2, 1])?;
        let gated_z = {
            let one = scalar(1.0)?;
            let neg = zc.neg()?;
            let exp = neg.exp()?;
            let denom = exp.add(&one)?;
            zc.div(&denom)?
        };
        let chunk_out = normalized.mul(&gated_z)?;
        chunks.push(chunk_out);

        start = end;
    }

    if chunks.len() == 1 {
        return Ok(chunks.pop().expect("one chunk"));
    }
    let refs: Vec<&TracedTensor> = chunks.iter().collect();
    TracedTensor::concatenate(&refs, 2)
}

/// Full graph-compiled Gated DeltaNet layer: `x (hidden, length)` and
/// `mask (length,)` traced inputs, returns `(hidden, length)`.
pub fn delta_layer_traced(
    cfg: &GatedDeltaConfig,
    weights: &GatedDeltaTracedWeights,
    x: &TracedTensor,
    mask: &TracedTensor,
    length: usize,
) -> R<TracedTensor> {
    let key_dim = cfg.key_dim;
    let value_dim = cfg.value_dim;
    let heads = cfg.value_heads;
    let key_width = key_dim * cfg.key_heads;
    let value_width = value_dim * cfg.value_heads;
    let conv_channels = 2 * key_width + value_width;
    let groups = cfg.value_heads / cfg.key_heads;

    let x_masked = x.mul(mask)?;

    let qkv = linear_col(&x_masked, &weights.qkv)?;
    let mixed = causal_depthwise_silu(&qkv, &weights.conv, conv_channels, length, cfg.conv_taps)?;
    let z = linear_col(&x_masked, &weights.z)?;
    let a = linear_col(&x_masked, &weights.a)?;
    let b = linear_col(&x_masked, &weights.b)?;

    let q_block = slice_2d(&mixed, 0, key_width, length)?;
    let q = q_block.reshape(vec![key_dim, cfg.key_heads, length])?;
    let q = q.transpose(&[1, 0, 2])?;
    let k_block = slice_2d(&mixed, key_width, key_width, length)?;
    let k = k_block.reshape(vec![key_dim, cfg.key_heads, length])?;
    let k = k.transpose(&[1, 0, 2])?;
    let expand = |block: TracedTensor| -> R<TracedTensor> {
        if groups == 1 {
            return Ok(block);
        }
        let broadcast =
            block.broadcast_in_dim(&[cfg.key_heads, groups, key_dim, length], &[0, 2, 3])?;
        let reordered = broadcast.transpose(&[1, 0, 2, 3])?;
        reordered.reshape(vec![heads, key_dim, length])
    };
    let q = expand(q)?;
    let k = expand(k)?;

    let v_block = slice_2d(&mixed, 2 * key_width, value_width, length)?;
    let v = v_block.reshape(vec![value_dim, heads, length])?;
    let v = v.transpose(&[1, 0, 2])?;
    let z = z.reshape(vec![value_dim, heads, length])?;
    let z = z.transpose(&[1, 0, 2])?;

    let q = l2_normalize_heads(&q, heads, length, 1e-6)?;
    let inv_scale = 1.0 / (key_dim as f64).sqrt();
    let q = q.scale_real(inv_scale)?;
    let k = l2_normalize_heads(&k, heads, length, 1e-6)?;

    let beta = {
        let one = scalar(1.0)?;
        let neg = b.neg()?;
        let exp = neg.exp()?;
        let denom = exp.add(&one)?;
        one.div(&denom)?
    };
    let dt_bias = weights.dt_bias.reshape(vec![heads, 1])?;
    let a_shifted = a.add(&dt_bias)?;
    let softplus = softplus(&a_shifted)?;
    let a_decay = weights.a_decay.reshape(vec![heads, 1])?;
    let decay = a_decay.mul(&softplus)?;

    let out = delta_scan_chunked(
        key_dim,
        value_dim,
        length,
        cfg.chunk_size,
        heads,
        &q,
        &k,
        &v,
        &z,
        &beta,
        &decay,
        &weights.norm,
        cfg.eps as f64,
    )?;

    let out = out.transpose(&[1, 0, 2])?;
    let out = out.reshape(vec![value_width, length])?;
    linear_col(&out, &weights.out_proj)
}

/// Build a `(hidden, length)` F32 traced input placeholder.
pub fn input_x(hidden: usize, length: usize) -> R<TracedTensor> {
    TracedTensor::input_concrete_shape(DType::F32, &[hidden, length])
}

/// Build a `(length,)` F32 traced input placeholder for the mask.
pub fn input_mask(length: usize) -> R<TracedTensor> {
    TracedTensor::input_concrete_shape(DType::F32, &[length])
}
