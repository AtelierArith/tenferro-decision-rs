//! Fully tensor-native Gated DeltaNet layer.
//!
//! Unlike [`crate::layer::delta_layer_tenferro`], which speaks host slices and
//! round-trips every intermediate through the host, this module keeps the whole
//! layer inside the eager session: masking, the fused causal depthwise
//! convolution, the Q/K L2 normalization, the decay gates, the chunked scan,
//! and the output projection are all composed from tenferro ops. The host path
//! ([`crate::layer::delta_layer_reference`]) stays the correctness oracle.
//!
//! The scan is **head-batched**: every value head runs as a leading batch axis
//! of a single chunk loop, so the eager-op count does not grow with the number
//! of heads. Because the layer is tensor-in/tensor-out it composes with the rest
//! of a tenferro forward (no `duplicate_value` between layers) and is
//! backend-portable.

use tenferro_ad::{
    DotGeneralConfig, EagerSession, EagerTensor, PadConfig, Result as AdResult, SliceConfig,
};
use tenferro_infer::{TensorCache, activation, norm};
use tenferro_linalg::EagerSessionLinalgExt;

use crate::chunked::constant;
use crate::config::GatedDeltaConfig;
use crate::layer::{GatedDeltaWeights, invalid_weights};

/// Prepared tensor weights for one Gated DeltaNet layer.
///
/// Build with [`prepare_tensor_weights`]; the tensors are cached by host storage
/// identity, so preparing the same layer again is a few `Arc` clones.
#[derive(Clone, Debug)]
pub struct GatedDeltaTensorWeights {
    /// Fused q/k/v projection `(hidden, 2*key_width + value_width)`.
    pub qkv: EagerTensor,
    /// Output gate projection `(hidden, value_width)`.
    pub z: EagerTensor,
    /// Decay projection `(hidden, value_heads)`.
    pub a: EagerTensor,
    /// Write-strength projection `(hidden, value_heads)`.
    pub b: EagerTensor,
    /// Depthwise convolution kernel `(conv_channels, conv_taps)`.
    pub conv: EagerTensor,
    /// `-exp(A_log)` `(value_heads,)`.
    pub a_decay: EagerTensor,
    /// Decay bias `(value_heads,)`.
    pub dt_bias: EagerTensor,
    /// Non-centered RMSNorm scale `(value_dim,)`.
    pub norm: EagerTensor,
    /// Output projection `(value_width, hidden)`.
    pub out_proj: EagerTensor,
}

/// Build the tensor weights for a layer, reusing `cache` across forwards.
pub fn prepare_tensor_weights(
    session: &mut EagerSession<'_>,
    cfg: &GatedDeltaConfig,
    weights: &GatedDeltaWeights,
    cache: &mut TensorCache,
) -> AdResult<GatedDeltaTensorWeights> {
    weights.validate(cfg).map_err(invalid_weights)?;
    let key_width = cfg.key_dim * cfg.key_heads;
    let value_width = cfg.value_dim * cfg.value_heads;
    let conv_channels = 2 * key_width + value_width;
    Ok(GatedDeltaTensorWeights {
        qkv: cache.col(session, vec![cfg.hidden, conv_channels], &weights.qkv)?,
        z: cache.col(session, vec![cfg.hidden, value_width], &weights.z)?,
        a: cache.col(session, vec![cfg.hidden, cfg.value_heads], &weights.a)?,
        b: cache.col(session, vec![cfg.hidden, cfg.value_heads], &weights.b)?,
        // Row-major `(taps, channels)` is already column-major `(channels, taps)`.
        conv: cache.col_major(session, vec![conv_channels, cfg.conv_taps], &weights.conv)?,
        a_decay: cache.col_major(session, vec![cfg.value_heads], &weights.a_decay)?,
        dt_bias: cache.col_major(session, vec![cfg.value_heads], &weights.dt_bias)?,
        norm: cache.col_major(session, vec![cfg.value_dim], &weights.norm)?,
        out_proj: cache.col(session, vec![value_width, cfg.hidden], &weights.out_proj)?,
    })
}

/// Prepared weights for the fused host recurrent kernel, in its **row-major**
/// layout, held as column-major tenferro tensors so their buffers *are* the
/// kernel operands (zero transpose). Build with [`prepare_kernel_weights`].
#[derive(Clone, Debug)]
pub struct GatedDeltaKernelWeights {
    /// `qkv (2k+v, hidden)`, buffer is row-major `(hidden, 2k+v)`.
    pub qkv: EagerTensor,
    /// `z (v, hidden)`, buffer is row-major `(hidden, v)`.
    pub z: EagerTensor,
    /// `a (value_heads, hidden)`.
    pub a: EagerTensor,
    /// `b (value_heads, hidden)`.
    pub b: EagerTensor,
    /// `conv (2k+v, conv_taps)`, buffer is row-major `(conv_taps, 2k+v)`.
    pub conv: EagerTensor,
    /// `a_decay (value_heads,)`.
    pub a_decay: EagerTensor,
    /// `dt_bias (value_heads,)`.
    pub dt_bias: EagerTensor,
    /// `norm (value_dim,)`.
    pub norm: EagerTensor,
    /// `out_proj (hidden, v)`, buffer is row-major `(v, hidden)`.
    pub out_proj: EagerTensor,
}

/// Build the [`GatedDeltaKernelWeights`] for a layer, reusing `cache`.
///
/// Unlike [`prepare_tensor_weights`], these tensors keep the host kernel's
/// row-major buffers verbatim (their logical shapes are transposed), so the
/// `GatedDelta` extension op can read them in place.
pub fn prepare_kernel_weights(
    session: &mut EagerSession<'_>,
    cfg: &GatedDeltaConfig,
    weights: &GatedDeltaWeights,
    cache: &mut TensorCache,
) -> AdResult<GatedDeltaKernelWeights> {
    weights.validate(cfg).map_err(invalid_weights)?;
    let key_width = cfg.key_dim * cfg.key_heads;
    let value_width = cfg.value_dim * cfg.value_heads;
    let conv_channels = 2 * key_width + value_width;
    Ok(GatedDeltaKernelWeights {
        qkv: cache.col_major(session, vec![conv_channels, cfg.hidden], &weights.qkv)?,
        z: cache.col_major(session, vec![value_width, cfg.hidden], &weights.z)?,
        a: cache.col_major(session, vec![cfg.value_heads, cfg.hidden], &weights.a)?,
        b: cache.col_major(session, vec![cfg.value_heads, cfg.hidden], &weights.b)?,
        conv: cache.col_major(session, vec![conv_channels, cfg.conv_taps], &weights.conv)?,
        a_decay: cache.col_major(session, vec![cfg.value_heads], &weights.a_decay)?,
        dt_bias: cache.col_major(session, vec![cfg.value_heads], &weights.dt_bias)?,
        norm: cache.col_major(session, vec![cfg.value_dim], &weights.norm)?,
        out_proj: cache.col_major(session, vec![cfg.hidden, value_width], &weights.out_proj)?,
    })
}

/// A rank-1 constant, used for broadcast scalars.
fn scalar(session: &mut EagerSession<'_>, value: f32) -> AdResult<EagerTensor> {
    constant(session, &[1], &[value])
}

/// `y = weightᵀ x` for `x (in, L)` and prepared `weight (in, out)`.
pub(crate) fn linear_col(
    session: &mut EagerSession<'_>,
    x: &EagerTensor,
    weight: &EagerTensor,
) -> AdResult<EagerTensor> {
    let y = session.dot_general(
        x,
        weight,
        DotGeneralConfig {
            lhs_contracting_dims: [0].as_slice().into(),
            rhs_contracting_dims: [0].as_slice().into(),
            lhs_batch_dims: [].as_slice().into(),
            rhs_batch_dims: [].as_slice().into(),
        },
    )?;
    session.transpose(&y, &[1, 0])
}

/// Batched `a (H, m, k) @ b (H, k, n) -> (H, m, n)`.
///
/// tenferro's `dot_general` emits the batch dimensions **trailing**
/// (`(m, n, H)` here), so the result is moved back to batch-leading.
fn bmm(session: &mut EagerSession<'_>, a: &EagerTensor, b: &EagerTensor) -> AdResult<EagerTensor> {
    let out = session.dot_general(
        a,
        b,
        DotGeneralConfig {
            lhs_contracting_dims: [2].as_slice().into(),
            rhs_contracting_dims: [1].as_slice().into(),
            lhs_batch_dims: [0].as_slice().into(),
            rhs_batch_dims: [0].as_slice().into(),
        },
    )?;
    session.transpose(&out, &[2, 0, 1])
}

fn slice_rows(
    session: &mut EagerSession<'_>,
    input: &EagerTensor,
    row_start: usize,
    rows: usize,
    cols: usize,
) -> AdResult<EagerTensor> {
    session.slice(
        input,
        SliceConfig {
            starts: vec![row_start, 0],
            limits: vec![row_start + rows, cols],
            strides: vec![1, 1],
        },
    )
}

/// Slice the trailing (length) axis of a `(rows, length)` tensor.
fn slice_time_rows(
    session: &mut EagerSession<'_>,
    input: &EagerTensor,
    rows: usize,
    start: usize,
    end: usize,
) -> AdResult<EagerTensor> {
    session.slice(
        input,
        SliceConfig {
            starts: vec![0, start],
            limits: vec![rows, end],
            strides: vec![1, 1],
        },
    )
}

/// Slice the trailing (length) axis of a `(outer, rows, length)` tensor.
fn slice_time(
    session: &mut EagerSession<'_>,
    input: &EagerTensor,
    outer: usize,
    rows: usize,
    start: usize,
    end: usize,
) -> AdResult<EagerTensor> {
    session.slice(
        input,
        SliceConfig {
            starts: vec![0, 0, start],
            limits: vec![outer, rows, end],
            strides: vec![1, 1, 1],
        },
    )
}

/// `softplus(x) = max(x, 0) + log(1 + exp(-|x|))`, matching [`crate::ops`].
fn softplus(session: &mut EagerSession<'_>, x: &EagerTensor) -> AdResult<EagerTensor> {
    let zero = scalar(session, 0.0)?;
    let one = scalar(session, 1.0)?;
    let positive = session.maximum(x, &zero)?;
    let abs = session.abs(x)?;
    let neg_abs = session.scale_real(&abs, -1.0)?;
    let exp = session.exp(&neg_abs)?;
    let shifted = session.add(&exp, &one)?;
    let log = session.log(&shifted)?;
    session.add(&positive, &log)
}

/// Causal depthwise convolution with a fused SiLU, all in the session.
///
/// `x` is `(channels, length)`, `conv` is `(channels, taps)`. Tap `tap` has lag
/// `taps - 1 - tap`; each shift is a zero-pad on the left followed by a slice.
fn causal_depthwise_silu_tensor(
    session: &mut EagerSession<'_>,
    x: &EagerTensor,
    conv: &EagerTensor,
    channels: usize,
    length: usize,
    taps: usize,
) -> AdResult<EagerTensor> {
    let mut acc: Option<EagerTensor> = None;
    for tap in 0..taps {
        let lag = taps - 1 - tap;
        let shifted = if lag == 0 {
            x.clone()
        } else {
            let padded = session.pad(
                x,
                PadConfig {
                    edge_padding_low: vec![0, lag as i64],
                    edge_padding_high: vec![0, 0],
                    interior_padding: vec![0, 0],
                },
            )?;
            session.slice(
                &padded,
                SliceConfig {
                    starts: vec![0, 0],
                    limits: vec![channels, length],
                    strides: vec![1, 1],
                },
            )?
        };
        let weight = session.slice(
            conv,
            SliceConfig {
                starts: vec![0, tap],
                limits: vec![channels, tap + 1],
                strides: vec![1, 1],
            },
        )?;
        let term = session.mul(&shifted, &weight)?;
        acc = Some(match acc {
            None => term,
            Some(previous) => session.add(&previous, &term)?,
        });
    }
    activation::silu(session, &acc.expect("conv_taps >= 1"))
}

/// L2-normalize each column over the head axis of `x (heads, rows, length)`.
fn l2_normalize_heads(
    session: &mut EagerSession<'_>,
    x: &EagerTensor,
    heads: usize,
    length: usize,
    eps: f64,
) -> AdResult<EagerTensor> {
    let sum_sq = session.reduce_sum_squares(x, &[1])?; // (heads, length)
    let eps = session.constant_from_host(tenferro_ad::Tensor::from_vec_col_major(
        vec![length],
        vec![eps as f32; length],
    )?)?;
    let denom = session.add(&sum_sq, &eps)?;
    let inv = session.rsqrt(&denom)?;
    let inv = session.reshape(&inv, vec![heads, 1, length])?;
    session.mul(x, &inv)
}

/// Lower-triangular ones `(n, n)` (including the diagonal).
fn lower_tri_ones(session: &mut EagerSession<'_>, n: usize) -> AdResult<EagerTensor> {
    let ones = constant(session, &[n, n], &vec![1.0f32; n * n])?;
    session.tril(&ones, 0)
}

/// Extract head `h` of a `(heads, rows, cols)` tensor as `(rows, cols)`.
fn slice_head(
    session: &mut EagerSession<'_>,
    input: &EagerTensor,
    h: usize,
    rows: usize,
    cols: usize,
) -> AdResult<EagerTensor> {
    let view = session.slice(
        input,
        SliceConfig {
            starts: vec![h, 0, 0],
            limits: vec![h + 1, rows, cols],
            strides: vec![1, 1, 1],
        },
    )?;
    session.reshape(&view, vec![rows, cols])
}

/// Slice the feature axis of a `(heads, n, features)` tensor.
fn slice_features(
    session: &mut EagerSession<'_>,
    input: &EagerTensor,
    heads: usize,
    n: usize,
    start: usize,
    end: usize,
) -> AdResult<EagerTensor> {
    session.slice(
        input,
        SliceConfig {
            starts: vec![0, 0, start],
            limits: vec![heads, n, end],
            strides: vec![1, 1, 1],
        },
    )
}

/// Batched unit-diagonal lower-triangular solve `X = L \ B`.
///
/// `triangular_solve` is rank-2 only, so this loops over the batch axis and
/// stacks the results; every other op in the scan stays batched.
fn solve_unit_lower_heads(
    session: &mut EagerSession<'_>,
    system: &EagerTensor,
    rhs: &EagerTensor,
    heads: usize,
    n: usize,
    cols: usize,
) -> AdResult<EagerTensor> {
    let mut solved = Vec::with_capacity(heads);
    for h in 0..heads {
        let a = slice_head(session, system, h, n, n)?;
        let b = slice_head(session, rhs, h, n, cols)?;
        let x = session.triangular_solve(&a, &b, true, true, false, true)?;
        solved.push(session.reshape(&x, vec![1, n, cols])?);
    }
    let refs: Vec<&EagerTensor> = solved.iter().collect();
    session.concatenate(&refs, 0)
}

/// Head-batched tensor-native chunked scan.
///
/// `q`/`k` are `(heads, key_dim, length)`, `v`/`z` `(heads, value_dim,
/// length)`, `beta`/`decay` `(heads, length)`. Returns `(heads, value_dim,
/// length)`. Mirrors [`crate::chunked::delta_scan_chunked`] with every value
/// head on a leading batch axis.
#[allow(clippy::too_many_arguments)]
fn delta_scan_chunked_batched(
    session: &mut EagerSession<'_>,
    key_dim: usize,
    value_dim: usize,
    length: usize,
    chunk_size: usize,
    heads: usize,
    q: &EagerTensor,
    k: &EagerTensor,
    v: &EagerTensor,
    z: &EagerTensor,
    beta: &EagerTensor,
    decay: &EagerTensor,
    norm_weight: &EagerTensor,
    eps: f64,
) -> AdResult<EagerTensor> {
    let neg_big = scalar(session, -1.0e30)?;
    let zero = scalar(session, 0.0)?;
    let mut state = constant(
        session,
        &[heads, value_dim, key_dim],
        &vec![0.0f32; heads * value_dim * key_dim],
    )?;
    let mut chunks: Vec<EagerTensor> = Vec::new();

    let mut start = 0usize;
    while start < length {
        let end = (start + chunk_size).min(length);
        let n = end - start;

        let qc = slice_time(session, q, heads, key_dim, start, end)?;
        let kc = slice_time(session, k, heads, key_dim, start, end)?;
        let vc = slice_time(session, v, heads, value_dim, start, end)?;
        let zc = slice_time(session, z, heads, value_dim, start, end)?;
        let bc = slice_time_rows(session, beta, heads, start, end)?; // (H, n)
        let dc = slice_time_rows(session, decay, heads, start, end)?; // (H, n)

        // Cumulative log-decay via a lower-tri ones matmul, plus the pair decay.
        let lower = lower_tri_ones(session, n)?;
        let lower = session.broadcast_in_dim(&lower, &[heads, n, n], &[1, 2])?;
        let dc_col = session.reshape(&dc, vec![heads, n, 1])?;
        let cumulative = bmm(session, &lower, &dc_col)?; // (H, n, 1)
        let cumulative = session.reshape(&cumulative, vec![heads, n])?;
        let exp_decay = session.exp(&cumulative)?; // (H, n)

        let cum_col = session.reshape(&cumulative, vec![heads, n, 1])?;
        let cum_row = session.reshape(&cumulative, vec![heads, 1, n])?;
        let diff = session.sub(&cum_col, &cum_row)?; // (H, n, n)
        let diff = session.clamp(&diff, &neg_big, &zero)?;
        let pair = session.exp(&diff)?;
        let pair_decay = session.mul(&pair, &lower)?;
        let pair_decay_t = session.transpose(&pair_decay, &[0, 2, 1])?;

        // system = (kc * beta)^T @ kc .* pair_decay
        let bc_k = session.reshape(&bc, vec![heads, 1, n])?;
        let kc_beta = session.mul(&kc, &bc_k)?; // (H, kd, n)
        let kc_beta_t = session.transpose(&kc_beta, &[0, 2, 1])?;
        let system = bmm(session, &kc_beta_t, &kc)?;
        let system = session.mul(&system, &pair_decay)?;

        let vc_beta = session.mul(&vc, &bc_k)?;
        let rhs_values = session.transpose(&vc_beta, &[0, 2, 1])?;

        let e_k = session.reshape(&exp_decay, vec![heads, 1, n])?;
        let kc_beta_e = session.mul(&kc_beta, &e_k)?;
        let rhs_keys = session.transpose(&kc_beta_e, &[0, 2, 1])?;

        // Solve for the values and keys in one batched pass (the system is the
        // same): stack their right-hand sides along the feature axis.
        let rhs = session.concatenate(&[&rhs_values, &rhs_keys], 2)?;
        let solved = solve_unit_lower_heads(session, &system, &rhs, heads, n, value_dim + key_dim)?;
        let new_values = slice_features(session, &solved, heads, n, 0, value_dim)?;
        let reading_keys =
            slice_features(session, &solved, heads, n, value_dim, value_dim + key_dim)?;

        // corrections = new_values^T - state @ reading_keys^T
        let new_values_t = session.transpose(&new_values, &[0, 2, 1])?;
        let reading_keys_t = session.transpose(&reading_keys, &[0, 2, 1])?;
        let state_reading = bmm(session, &state, &reading_keys_t)?;
        let corrections = session.sub(&new_values_t, &state_reading)?;

        // intra = (kc^T @ qc) .* pair_decay^T
        let kc_t = session.transpose(&kc, &[0, 2, 1])?;
        let kq = bmm(session, &kc_t, &qc)?;
        let intra = session.mul(&kq, &pair_decay_t)?;

        // result = state @ (qc .* exp_decay) + corrections @ intra
        let qc_e = session.mul(&qc, &e_k)?;
        let state_q = bmm(session, &state, &qc_e)?;
        let correction_intra = bmm(session, &corrections, &intra)?;
        let result = session.add(&state_q, &correction_intra)?;

        // state update: tail[j] = exp(cum_last - cum[j]).
        let cum_last = slice_time_rows(session, &cumulative, heads, n - 1, n)?; // (H, 1)
        let tail = session.sub(&cum_last, &cumulative)?; // (H, 1) - (H, n)
        let tail = session.exp(&tail)?;
        let tail_k = session.reshape(&tail, vec![heads, 1, n])?;
        let final_decay = session.exp(&cum_last)?;
        let final_decay = session.reshape(&final_decay, vec![heads, 1, 1])?;
        let ending_keys = session.mul(&kc, &tail_k)?;
        let ending_keys_t = session.transpose(&ending_keys, &[0, 2, 1])?;
        let state_scaled = session.mul(&state, &final_decay)?;
        let correction_keys = bmm(session, &corrections, &ending_keys_t)?;
        state = session.add(&state_scaled, &correction_keys)?;

        // Non-centered RMSNorm over the value dimension, then the output gate.
        let result_t = session.transpose(&result, &[0, 2, 1])?; // (H, n, vd)
        let normalized_t = norm::rms_norm(session, &result_t, norm_weight, false, eps)?;
        let normalized = session.transpose(&normalized_t, &[0, 2, 1])?;
        let gated_z = activation::silu(session, &zc)?;
        let chunk_out = session.mul(&normalized, &gated_z)?;
        chunks.push(chunk_out);

        start = end;
    }

    let refs: Vec<&EagerTensor> = chunks.iter().collect();
    session.concatenate(&refs, 2)
}

/// Full tensor-native Gated DeltaNet layer.
///
/// `x` is `(hidden, length)`; `mask` is the host attention mask. Returns
/// `(hidden, length)`, all inside the session.
pub fn delta_layer_tenferro_native(
    session: &mut EagerSession<'_>,
    cfg: &GatedDeltaConfig,
    weights: &GatedDeltaTensorWeights,
    x: &EagerTensor,
    mask: &[f32],
) -> AdResult<EagerTensor> {
    let mask_t = constant(session, &[mask.len()], mask)?;
    delta_layer_tenferro_prepared_mask(session, cfg, weights, x, &mask_t)
}

/// Tensor-native layer accepting an already prepared attention mask.
///
/// `x` is `(hidden, length)` and `mask` is `(length,)`. Both may remain
/// device-resident across layers; this entry point never reads the mask on the
/// host. Mask values retain the host wrapper's multiplicative semantics.
pub fn delta_layer_tenferro_prepared_mask(
    session: &mut EagerSession<'_>,
    cfg: &GatedDeltaConfig,
    weights: &GatedDeltaTensorWeights,
    x: &EagerTensor,
    mask: &EagerTensor,
) -> AdResult<EagerTensor> {
    cfg.validate().map_err(invalid_weights)?;
    let mask_dims = mask.shape();
    if mask_dims.len() != 1 || mask_dims[0] == 0 || x.shape() != [cfg.hidden, mask_dims[0]] {
        return Err(tenferro_ad::Error::TensorRuntime(
            tenferro_tensor::Error::invalid_argument(
                "delta_layer_tenferro_prepared_mask",
                "shape",
                "GatedDelta requires x [hidden, length] and a nonempty mask [length]",
            ),
        ));
    }
    let length = mask_dims[0];
    let key_width = cfg.key_dim * cfg.key_heads;
    let value_width = cfg.value_dim * cfg.value_heads;
    let conv_channels = 2 * key_width + value_width;

    let x_masked = session.mul(x, mask)?;

    let qkv = linear_col(session, &x_masked, &weights.qkv)?; // (conv_channels, L)
    let mixed = causal_depthwise_silu_tensor(
        session,
        &qkv,
        &weights.conv,
        conv_channels,
        length,
        cfg.conv_taps,
    )?;
    let z = linear_col(session, &x_masked, &weights.z)?; // (value_width, L)
    let a = linear_col(session, &x_masked, &weights.a)?; // (value_heads, L)
    let b = linear_col(session, &x_masked, &weights.b)?; // (value_heads, L)

    delta_layer_from_projected(
        session,
        cfg,
        weights,
        ProjectedDeltaTensors {
            mixed: &mixed,
            z: &z,
            a: &a,
            b: &b,
        },
    )
}

/// Device-resident layer intermediates after projection and convolution/SiLU.
pub struct ProjectedDeltaTensors<'a> {
    /// Convolved Q/K/V, `[2 * key_width + value_width, length]`.
    pub mixed: &'a EagerTensor,
    /// Output gate projection, `[value_width, length]`.
    pub z: &'a EagerTensor,
    /// Decay projection, `[value_heads, length]`.
    pub a: &'a EagerTensor,
    /// Write gate projection, `[value_heads, length]`.
    pub b: &'a EagerTensor,
}

/// Complete a prepared layer using native normalization, chunked scan/readout.
///
/// The head-batched scan uses native triangular solves, including key widths
/// above the recurrent kernel limit. No intermediate tensor is read on host.
/// This also lets a CUDA adapter substitute raw convolution without duplicating
/// the backend-agnostic scan and head-grouping logic.
pub fn delta_layer_from_projected(
    session: &mut EagerSession<'_>,
    cfg: &GatedDeltaConfig,
    weights: &GatedDeltaTensorWeights,
    inputs: ProjectedDeltaTensors<'_>,
) -> AdResult<EagerTensor> {
    cfg.validate().map_err(invalid_weights)?;
    let invalid = || {
        tenferro_ad::Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
            "delta_layer_from_projected",
            "shape",
            "prepared projections must have matching nonempty length and configured widths",
        ))
    };
    let key_dim = cfg.key_dim;
    let value_dim = cfg.value_dim;
    let heads = cfg.value_heads;
    let key_width = key_dim.checked_mul(cfg.key_heads).ok_or_else(invalid)?;
    let value_width = value_dim.checked_mul(cfg.value_heads).ok_or_else(invalid)?;
    let expected_channels = key_width
        .checked_mul(2)
        .and_then(|n| n.checked_add(value_width))
        .ok_or_else(invalid)?;
    let groups = cfg.value_heads / cfg.key_heads;
    let ProjectedDeltaTensors { mixed, z, a, b } = inputs;
    let [channels, length] = mixed.shape() else {
        return Err(invalid());
    };
    let length = *length;
    if length == 0
        || *channels != expected_channels
        || z.shape() != [value_width, length]
        || a.shape() != [heads, length]
        || b.shape() != [heads, length]
    {
        return Err(invalid());
    }
    // Split Q/K/V/Z into heads. tenferro's `reshape` preserves column-major
    // order, so `[width, L]` splits to `(dim, heads, L)` and then transposes to
    // heads-leading (matching the attention code's `project_heads`).
    let q_block = slice_rows(session, mixed, 0, key_width, length)?;
    let q = session.reshape(&q_block, vec![key_dim, cfg.key_heads, length])?;
    let q = session.transpose(&q, &[1, 0, 2])?; // (key_heads, key_dim, L)
    let k_block = slice_rows(session, mixed, key_width, key_width, length)?;
    let k = session.reshape(&k_block, vec![key_dim, cfg.key_heads, length])?;
    let k = session.transpose(&k, &[1, 0, 2])?;
    // Expand the key heads to value heads (consecutive grouping: head `h`
    // serves value heads `h*groups .. h*groups+groups`).
    let expand = |session: &mut EagerSession<'_>, block: EagerTensor| -> AdResult<EagerTensor> {
        if groups == 1 {
            return Ok(block);
        }
        let broadcast = session.broadcast_in_dim(
            &block,
            &[cfg.key_heads, groups, key_dim, length],
            &[0, 2, 3],
        )?;
        let reordered = session.transpose(&broadcast, &[1, 0, 2, 3])?;
        session.reshape(&reordered, vec![heads, key_dim, length])
    };
    let q = expand(session, q)?;
    let k = expand(session, k)?;

    let v_block = slice_rows(session, mixed, 2 * key_width, value_width, length)?;
    let v = session.reshape(&v_block, vec![value_dim, heads, length])?;
    let v = session.transpose(&v, &[1, 0, 2])?;
    let z = session.reshape(z, vec![value_dim, heads, length])?;
    let z = session.transpose(&z, &[1, 0, 2])?;

    // L2-normalize Q/K over the head width (eps fixed at 1e-6, as in the host
    // reference) and scale Q by 1/sqrt(key_dim).
    let q = l2_normalize_heads(session, &q, heads, length, 1e-6)?;
    let inv_scale = 1.0 / (key_dim as f64).sqrt();
    let q = session.scale_real(&q, inv_scale)?;
    let k = l2_normalize_heads(session, &k, heads, length, 1e-6)?;

    // Gates: beta = sigmoid(b); decay = a_decay * softplus(a + dt_bias).
    let beta = activation::sigmoid(session, b)?; // (heads, L)
    let dt_bias = session.reshape(&weights.dt_bias, vec![heads, 1])?;
    let a_shifted = session.add(a, &dt_bias)?;
    let softplus = softplus(session, &a_shifted)?;
    let a_decay = session.reshape(&weights.a_decay, vec![heads, 1])?;
    let decay = session.mul(&a_decay, &softplus)?;

    let out = delta_scan_chunked_batched(
        session,
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
    )?; // (heads, value_dim, L)

    let out = session.transpose(&out, &[1, 0, 2])?; // (value_dim, heads, L)
    let out = session.reshape(&out, vec![value_width, length])?;
    linear_col(session, &out, &weights.out_proj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunked::col_major;
    use crate::layer::delta_layer_reference;

    fn lcg(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*state >> 40) as f32) / (1u64 << 24) as f32 - 0.5
    }

    fn runtime() -> std::sync::Arc<tenferro_ad::EagerRuntime> {
        tenferro_ad::EagerRuntime::with_cpu_backend(tenferro_cpu::CpuBackend::new()).unwrap()
    }

    #[test]
    fn conv_matches_host() {
        let channels = 6usize;
        let length = 9usize;
        let taps = 4usize;
        let mut s = 7u64;
        let x: Vec<f32> = (0..channels * length).map(|_| lcg(&mut s)).collect();
        let w: Vec<f32> = (0..taps * channels).map(|_| lcg(&mut s)).collect();
        let want = crate::conv::causal_depthwise_silu(&x, channels, length, &w, taps);
        let got = runtime()
            .with_eager_session(|session| {
                let xt = constant(
                    session,
                    &[channels, length],
                    &col_major(channels, length, &x)?,
                )?;
                let wt = session.constant_from(tenferro_ad::Tensor::from_vec_col_major(
                    vec![channels, taps],
                    w.clone(),
                )?)?;
                let out = causal_depthwise_silu_tensor(session, &xt, &wt, channels, length, taps)?;
                let host = session.duplicate_value(&out)?;
                let values = host.as_slice::<f32>()?;
                Ok::<Vec<f32>, tenferro_ad::Error>(
                    (0..channels * length)
                        .map(|i| values[(i / length) + (i % length) * channels])
                        .collect(),
                )
            })
            .unwrap()
            .unwrap();
        let diff = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff < 1e-5, "conv max diff {diff}");
    }

    #[test]
    fn native_matches_reference() {
        let cfg = GatedDeltaConfig {
            hidden: 8,
            key_dim: 4,
            value_dim: 4,
            key_heads: 2,
            value_heads: 4,
            conv_taps: 4,
            chunk_size: 3,
            eps: 1e-5,
            algorithm: crate::config::Algorithm::Chunked,
        };
        let key_width = cfg.key_dim * cfg.key_heads;
        let value_width = cfg.value_dim * cfg.value_heads;
        let conv_channels = 2 * key_width + value_width;
        let mut state = 424242u64;
        fn rand_vec(state: &mut u64, n: usize) -> Vec<f32> {
            (0..n).map(|_| lcg(state)).collect()
        }
        let weights = GatedDeltaWeights {
            qkv: rand_vec(&mut state, cfg.hidden * conv_channels),
            z: rand_vec(&mut state, cfg.hidden * value_width),
            a: rand_vec(&mut state, cfg.hidden * cfg.value_heads),
            b: rand_vec(&mut state, cfg.hidden * cfg.value_heads),
            conv: rand_vec(&mut state, cfg.conv_taps * conv_channels),
            a_decay: (0..cfg.value_heads)
                .map(|_| -0.1 - lcg(&mut state).abs())
                .collect(),
            dt_bias: rand_vec(&mut state, cfg.value_heads),
            norm: (0..cfg.value_dim)
                .map(|_| 0.5 + lcg(&mut state).abs())
                .collect(),
            out_proj: rand_vec(&mut state, value_width * cfg.hidden),
        };
        let length = 7usize;
        let x = rand_vec(&mut state, cfg.hidden * length);
        let mask = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];

        let reference = delta_layer_reference(&cfg, &weights, &x, &mask).unwrap();
        let native = runtime()
            .with_eager_session(|session| {
                let mut cache = TensorCache::new();
                let tw = prepare_tensor_weights(session, &cfg, &weights, &mut cache)?;
                let x = constant(
                    session,
                    &[cfg.hidden, length],
                    &col_major(cfg.hidden, length, &x)?,
                )?;
                let mask = constant(session, &[length], &mask)?;
                let out = delta_layer_tenferro_prepared_mask(session, &cfg, &tw, &x, &mask)?;
                let host = session.duplicate_value(&out)?;
                let values = host.as_slice::<f32>()?;
                // col-major (hidden, length) -> row-major
                Ok::<Vec<f32>, tenferro_ad::Error>(
                    (0..cfg.hidden * length)
                        .map(|index| {
                            let row = index / length;
                            let col = index % length;
                            values[row + col * cfg.hidden]
                        })
                        .collect(),
                )
            })
            .unwrap()
            .unwrap();

        let diff = reference
            .iter()
            .zip(&native)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff < 1e-3, "native vs reference max diff {diff}");
    }
}
