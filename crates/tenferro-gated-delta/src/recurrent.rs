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

use rayon::prelude::*;

use crate::config::GatedDeltaConfig;
use crate::conv::causal_depthwise_silu_into;
use crate::layer::{GatedDeltaWeightSlices, GatedDeltaWeights, linear_into, mask_rows_into};
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
    weights.validate(cfg)?;
    let length = mask.len();
    let kd = cfg.key_dim;
    let vd = cfg.value_dim;
    let key_width = kd * cfg.key_heads;
    let value_width = vd * cfg.value_heads;
    let conv_channels = 2 * key_width + value_width;

    ws.masked.resize(cfg.hidden * length, 0.0);
    mask_rows_into(x, cfg.hidden, length, mask, &mut ws.masked);

    ws.qkv_proj.resize(conv_channels * length, 0.0);
    linear_into(
        weights.qkv,
        cfg.hidden,
        conv_channels,
        &ws.masked,
        length,
        &mut ws.qkv_proj,
    );

    ws.mixed.resize(conv_channels * length, 0.0);
    causal_depthwise_silu_into(
        &ws.qkv_proj,
        conv_channels,
        length,
        weights.conv,
        cfg.conv_taps,
        &mut ws.mixed,
    );

    ws.z_proj.resize(value_width * length, 0.0);
    linear_into(
        weights.z,
        cfg.hidden,
        value_width,
        &ws.masked,
        length,
        &mut ws.z_proj,
    );

    ws.a_proj.resize(cfg.value_heads * length, 0.0);
    linear_into(
        weights.a,
        cfg.hidden,
        cfg.value_heads,
        &ws.masked,
        length,
        &mut ws.a_proj,
    );

    ws.b_proj.resize(cfg.value_heads * length, 0.0);
    linear_into(
        weights.b,
        cfg.hidden,
        cfg.value_heads,
        &ws.masked,
        length,
        &mut ws.b_proj,
    );

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

    // The value heads are independent: run them in parallel, each writing its
    // own `HeadScratch`.
    ws.heads.resize_with(cfg.value_heads, HeadScratch::default);
    let inv_scale = 1.0 / (kd as f32).sqrt();
    {
        let mixed = &ws.mixed;
        let z_proj = &ws.z_proj;
        let beta = &ws.beta;
        let decay = &ws.decay;
        ws.heads.par_iter_mut().enumerate().for_each(|(head, s)| {
            scan_head(
                cfg, weights, mixed, z_proj, beta, decay, length, head, inv_scale, s,
            );
        });
    }

    ws.out.resize(value_width * length, 0.0);
    {
        let heads = &ws.heads;
        let out = &mut ws.out;
        for (head, scratch) in heads.iter().enumerate() {
            let start = head * vd * length;
            out[start..start + vd * length].copy_from_slice(&scratch.output);
        }
    }

    ws.output.resize(cfg.hidden * length, 0.0);
    linear_into(
        weights.out_proj,
        value_width,
        cfg.hidden,
        &ws.out,
        length,
        &mut ws.output,
    );
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

    s.state.resize(vd * kd, 0.0);
    s.prediction.resize(vd, 0.0);
    s.correction.resize(vd, 0.0);
    s.result.resize(vd, 0.0);
    s.state.iter_mut().for_each(|value| *value = 0.0);

    for t in 0..length {
        let factor = decay[head * length + t].exp();
        let write = beta[head * length + t];

        for v in 0..vd {
            let mut acc = 0.0f32;
            for d in 0..kd {
                acc += s.state[v * kd + d] * s.k[d * length + t];
            }
            s.prediction[v] = acc;
        }
        for v in 0..vd {
            s.correction[v] = write * (s.v[v * length + t] - factor * s.prediction[v]);
        }
        for v in 0..vd {
            for d in 0..kd {
                s.state[v * kd + d] =
                    factor * s.state[v * kd + d] + s.correction[v] * s.k[d * length + t];
            }
        }
        for v in 0..vd {
            let mut acc = 0.0f32;
            for d in 0..kd {
                acc += s.state[v * kd + d] * s.q[d * length + t];
            }
            s.result[v] = acc;
        }

        rms_noncentered_in_place(&mut s.result, weights.norm, cfg.eps);
        for v in 0..vd {
            s.output[v * length + t] = s.result[v] * silu(s.z[v * length + t]);
        }
    }
}
