//! Fused host recurrent Gated DeltaNet layer.
//!
//! This is the optimized CPU formulation (design
//! `12_TENFERRO_GATED_DELTA.md` §9): projections, the causal convolution, Q/K
//! normalization, the gates, the recurrent scan, the output RMSNorm/gate, and
//! the output projection all run in host loops over one set of reusable
//! workspace buffers — no tenferro round-trips and no per-head intermediate
//! `Vec` allocations. The value heads share the layer projections but scan
//! independently, so the per-head work runs in parallel with `rayon`.
//!
//! It is numerically equivalent to [`crate::delta_layer_reference`] and is
//! cross-checked against it.

use pulp::{Simd, WithSimd};
use rayon::prelude::*;

use crate::config::GatedDeltaConfig;
use crate::conv::causal_depthwise_silu_into;
use crate::layer::{GatedDeltaWeightSlices, GatedDeltaWeights, mask_rows_into};
use crate::ops::{l2_normalize, rms_noncentered_in_place, sigmoid, silu, softplus};
use crate::workspace::{GatedDeltaWorkspace, HeadScratch};

/// Run the fused recurrent layer, returning a borrow of the workspace's output
/// buffer (`(hidden, length)` row-major).
pub fn delta_layer_recurrent<'a>(
    cfg: &GatedDeltaConfig,
    weights: &GatedDeltaWeights,
    x: &[f32],
    mask: &[f32],
    ws: &'a mut GatedDeltaWorkspace,
) -> decision_core::Result<&'a [f32]> {
    let slices = weights.slices();
    delta_layer_recurrent_slices(cfg, &slices, x, mask, ws)
}

/// [`delta_layer_recurrent`] on borrowed weight slices, so the fused kernel can
/// run directly on the tenferro extension op's in-place inputs (no copy).
pub fn delta_layer_recurrent_slices<'a>(
    cfg: &GatedDeltaConfig,
    weights: &GatedDeltaWeightSlices<'_>,
    x: &[f32],
    mask: &[f32],
    ws: &'a mut GatedDeltaWorkspace,
) -> decision_core::Result<&'a [f32]> {
    delta_layer_recurrent_with_projection(
        cfg,
        weights,
        x,
        mask,
        ws,
        &mut cpu_kernels::PortableProjection,
    )
}

/// The existing recurrent CPU layer with a prepared projection provider.
/// The scan and normalization are identical to the portable formulation.
pub fn delta_layer_recurrent_with_projection<'a>(
    cfg: &GatedDeltaConfig,
    weights: &GatedDeltaWeightSlices<'_>,
    x: &[f32],
    mask: &[f32],
    ws: &'a mut GatedDeltaWorkspace,
    projection: &mut dyn cpu_kernels::RowMajorProjection,
) -> decision_core::Result<&'a [f32]> {
    weights.validate(cfg)?;
    let length = mask.len();
    let kd = cfg.key_dim;
    let vd = cfg.value_dim;
    let key_width = kd * cfg.key_heads;
    let value_width = vd * cfg.value_heads;
    let conv_channels = 2 * key_width + value_width;

    // Optional stage profiling, enabled with `TENFERRO_DELTA_PROFILE=1`. The
    // env lookup is cached so the disabled path is a single atomic load.
    static DELTA_PROFILE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let profile =
        *DELTA_PROFILE.get_or_init(|| std::env::var_os("TENFERRO_DELTA_PROFILE").is_some());
    let mut lap = std::time::Instant::now();
    macro_rules! stage {
        ($name:literal) => {
            if profile {
                eprintln!(
                    "    delta {:<10} {:8.3} ms",
                    $name,
                    lap.elapsed().as_secs_f64() * 1e3
                );
                lap = std::time::Instant::now();
            }
        };
    }

    ws.masked.resize(cfg.hidden * length, 0.0);
    mask_rows_into(x, cfg.hidden, length, mask, &mut ws.masked);
    stage!("mask");

    ws.qkv_proj.resize(conv_channels * length, 0.0);
    projection
        .project(
            weights.qkv,
            cfg.hidden,
            conv_channels,
            &ws.masked,
            length,
            &mut ws.qkv_proj,
            false,
        )
        .map_err(|message| decision_core::DecisionError::Backend {
            message,
            source: None,
        })?;
    stage!("qkv");

    ws.mixed.resize(conv_channels * length, 0.0);
    causal_depthwise_silu_into(
        &ws.qkv_proj,
        conv_channels,
        length,
        weights.conv,
        cfg.conv_taps,
        &mut ws.mixed,
    );
    stage!("conv+silu");

    ws.z_proj.resize(value_width * length, 0.0);
    projection
        .project(
            weights.z,
            cfg.hidden,
            value_width,
            &ws.masked,
            length,
            &mut ws.z_proj,
            false,
        )
        .map_err(|message| decision_core::DecisionError::Backend {
            message,
            source: None,
        })?;
    stage!("z");

    ws.a_proj.resize(cfg.value_heads * length, 0.0);
    projection
        .project(
            weights.a,
            cfg.hidden,
            cfg.value_heads,
            &ws.masked,
            length,
            &mut ws.a_proj,
            false,
        )
        .map_err(|message| decision_core::DecisionError::Backend {
            message,
            source: None,
        })?;

    ws.b_proj.resize(cfg.value_heads * length, 0.0);
    projection
        .project(
            weights.b,
            cfg.hidden,
            cfg.value_heads,
            &ws.masked,
            length,
            &mut ws.b_proj,
            false,
        )
        .map_err(|message| decision_core::DecisionError::Backend {
            message,
            source: None,
        })?;
    stage!("ab");

    ws.beta.resize(cfg.value_heads * length, 0.0);
    ws.decay.resize(cfg.value_heads * length, 0.0);
    for head in 0..cfg.value_heads {
        for t in 0..length {
            let index = head * length + t;
            ws.beta[index] = sigmoid(ws.b_proj[index]);
            ws.decay[index] =
                weights.a_decay[head] * softplus(ws.a_proj[index] + weights.dt_bias[head]);
        }
    }
    stage!("beta/decay");

    // The value heads are independent: run them in parallel, each writing its
    // own `HeadScratch`.
    ws.heads.resize_with(cfg.value_heads, HeadScratch::default);
    let inv_scale = 1.0 / (kd as f32).sqrt();
    let arch = pulp::Arch::new();
    {
        let mixed = &ws.mixed;
        let z_proj = &ws.z_proj;
        let beta = &ws.beta;
        let decay = &ws.decay;
        ws.heads.par_iter_mut().enumerate().for_each(|(head, s)| {
            scan_head(
                cfg, weights, mixed, z_proj, beta, decay, length, head, inv_scale, s, arch,
            );
        });
    }
    stage!("scan");

    ws.out.resize(value_width * length, 0.0);
    {
        let heads = &ws.heads;
        let out = &mut ws.out;
        for (head, scratch) in heads.iter().enumerate() {
            let start = head * vd * length;
            out[start..start + vd * length].copy_from_slice(&scratch.output);
        }
    }
    stage!("out-copy");

    ws.output.resize(cfg.hidden * length, 0.0);
    projection
        .project(
            weights.out_proj,
            value_width,
            cfg.hidden,
            &ws.out,
            length,
            &mut ws.output,
            false,
        )
        .map_err(|message| decision_core::DecisionError::Backend {
            message,
            source: None,
        })?;
    stage!("out_proj");
    let _ = lap;
    Ok(&ws.output)
}

/// Extract, normalize, and scan one value head into `s.output`.
#[allow(clippy::too_many_arguments)]
fn scan_head(
    cfg: &GatedDeltaConfig,
    weights: &GatedDeltaWeightSlices<'_>,
    mixed: &[f32],
    z_proj: &[f32],
    beta: &[f32],
    decay: &[f32],
    length: usize,
    head: usize,
    inv_scale: f32,
    s: &mut HeadScratch,
    arch: pulp::Arch,
) {
    let kd = cfg.key_dim;
    let vd = cfg.value_dim;
    let key_width = kd * cfg.key_heads;
    let key_head = cfg.key_head_for(head);
    let q_offset = key_head * kd;
    let k_offset = key_width + key_head * kd;
    let v_offset = 2 * key_width + head * vd;
    let z_offset = head * vd;

    s.q.resize(kd * length, 0.0);
    s.k.resize(kd * length, 0.0);
    s.v.resize(vd * length, 0.0);
    s.z.resize(vd * length, 0.0);
    s.scratch.resize(kd.max(vd), 0.0);
    s.output.resize(vd * length, 0.0);

    for row in 0..kd {
        for t in 0..length {
            s.q[row * length + t] = mixed[(q_offset + row) * length + t];
            s.k[row * length + t] = mixed[(k_offset + row) * length + t];
        }
    }
    for row in 0..vd {
        for t in 0..length {
            s.v[row * length + t] = mixed[(v_offset + row) * length + t];
            s.z[row * length + t] = z_proj[(z_offset + row) * length + t];
        }
    }

    // L2-normalize Q and K per position; Q is additionally scaled by
    // `1 / sqrt(key_dim)` (see the layer reference for the orientation note).
    for t in 0..length {
        for d in 0..kd {
            s.scratch[d] = s.q[d * length + t];
        }
        l2_normalize(&mut s.scratch[..kd], 1e-6);
        for d in 0..kd {
            s.q[d * length + t] = s.scratch[d] * inv_scale;
        }
        for d in 0..kd {
            s.scratch[d] = s.k[d * length + t];
        }
        l2_normalize(&mut s.scratch[..kd], 1e-6);
        for d in 0..kd {
            s.k[d * length + t] = s.scratch[d];
        }
    }

    // The state S (value_dim × key_dim) is stored transposed as `(key_dim,
    // value_dim)`, so every scan loop is a contiguous axpy over the value
    // dimension with a scalar key/query — no stride-`length` gather. This
    // matches the Julia recurrent kernel and vectorizes; the summation order is
    // unchanged, so the result is bit-identical to the row-major scan.
    s.state.resize(vd * kd, 0.0);
    s.prediction.resize(vd, 0.0);
    s.correction.resize(vd, 0.0);
    s.result.resize(vd, 0.0);
    s.state.iter_mut().for_each(|value| *value = 0.0);

    arch.dispatch(ScanOp {
        state: &mut s.state,
        k: &s.k,
        q: &s.q,
        v_act: &s.v,
        z_act: &s.z,
        prediction: &mut s.prediction,
        correction: &mut s.correction,
        result: &mut s.result,
        output: &mut s.output,
        decay,
        beta,
        norm: weights.norm,
        head,
        kd,
        vd,
        length,
        eps: cfg.eps,
    });
}

/// The per-head recurrent token loop, written once and dispatched through
/// `pulp` so a baseline-compiled binary still uses AVX2/FMA (or whatever the
/// CPU supports) instead of relying on autovectorization.
///
/// The heavy inner loops (`S·k`, the fused `S ← factor·S + correction·kᵀ` /
/// `result = S·q`) run on the contiguous value axis; the per-token correction
/// gather and the strided output epilogue stay scalar.
struct ScanOp<'a> {
    state: &'a mut [f32],
    k: &'a [f32],
    q: &'a [f32],
    v_act: &'a [f32],
    z_act: &'a [f32],
    prediction: &'a mut [f32],
    correction: &'a mut [f32],
    result: &'a mut [f32],
    output: &'a mut [f32],
    decay: &'a [f32],
    beta: &'a [f32],
    norm: &'a [f32],
    head: usize,
    kd: usize,
    vd: usize,
    length: usize,
    eps: f32,
}

impl WithSimd for ScanOp<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let ScanOp {
            state,
            k,
            q,
            v_act,
            z_act,
            prediction,
            correction,
            result,
            output,
            decay,
            beta,
            norm,
            head,
            kd,
            vd,
            length,
            eps,
        } = self;

        for t in 0..length {
            let factor = decay[head * length + t].exp();
            let write = beta[head * length + t];

            // prediction = S · k.
            prediction.iter_mut().for_each(|value| *value = 0.0);
            for d in 0..kd {
                let key = k[d * length + t];
                let keys = simd.splat_f32s(key);
                let (row, row_tail) = S::as_simd_f32s(&state[d * vd..d * vd + vd]);
                let (pred, pred_tail) = S::as_mut_simd_f32s(prediction);
                for (p, s) in pred.iter_mut().zip(row.iter()) {
                    *p = simd.mul_add_f32s(keys, *s, *p);
                }
                for (p, s) in pred_tail.iter_mut().zip(row_tail.iter()) {
                    *p += *s * key;
                }
            }
            for (v, corr) in correction.iter_mut().enumerate() {
                *corr = write * (v_act[v * length + t] - factor * prediction[v]);
            }

            // Fused update and readout: S ← factor·S + correction·kᵀ, result = S·q.
            result.iter_mut().for_each(|value| *value = 0.0);
            let factors = simd.splat_f32s(factor);
            for d in 0..kd {
                let key = k[d * length + t];
                let query = q[d * length + t];
                let keys = simd.splat_f32s(key);
                let queries = simd.splat_f32s(query);
                let (row, row_tail) = S::as_mut_simd_f32s(&mut state[d * vd..d * vd + vd]);
                let (corr, corr_tail) = S::as_simd_f32s(correction);
                let (res, res_tail) = S::as_mut_simd_f32s(result);
                for ((cell, c), r) in row.iter_mut().zip(corr.iter()).zip(res.iter_mut()) {
                    let updated = simd.mul_add_f32s(keys, *c, simd.mul_f32s(factors, *cell));
                    *cell = updated;
                    *r = simd.mul_add_f32s(queries, updated, *r);
                }
                for ((cell, c), r) in row_tail
                    .iter_mut()
                    .zip(corr_tail.iter())
                    .zip(res_tail.iter_mut())
                {
                    let updated = factor * (*cell) + *c * key;
                    *cell = updated;
                    *r += updated * query;
                }
            }

            rms_noncentered_in_place(result, norm, eps);
            for v in 0..vd {
                output[v * length + t] = result[v] * silu(z_act[v * length + t]);
            }
        }
    }
}
