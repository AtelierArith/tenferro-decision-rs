//! Fused host recurrent Gated DeltaNet layer.
//!
//! This is the optimized CPU formulation (design
//! `12_TENFERRO_GATED_DELTA.md` §9): projections, the causal convolution, Q/K
//! normalization, the gates, the recurrent scan, the output RMSNorm/gate, and
//! the output projection all run in host loops over one set of reusable
//! workspace buffers — no tenferro round-trips and no per-head intermediate
//! `Vec` allocations. It is numerically equivalent to
//! [`crate::delta_layer_reference`] and is cross-checked against it.

use crate::config::GatedDeltaConfig;
use crate::conv::causal_depthwise_silu_into;
use crate::layer::{linear_into, mask_rows_into, GatedDeltaWeights};
use crate::ops::{l2_normalize, rms_noncentered_in_place, sigmoid, silu, softplus};
use crate::workspace::GatedDeltaWorkspace;

/// Run the fused recurrent layer, returning a borrow of the workspace's output
/// buffer (`(hidden, length)` row-major).
pub fn delta_layer_recurrent<'a>(
    cfg: &GatedDeltaConfig,
    weights: &GatedDeltaWeights,
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
        &weights.qkv,
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
        &weights.conv,
        cfg.conv_taps,
        &mut ws.mixed,
    );

    ws.z_proj.resize(value_width * length, 0.0);
    linear_into(
        &weights.z,
        cfg.hidden,
        value_width,
        &ws.masked,
        length,
        &mut ws.z_proj,
    );

    ws.a_proj.resize(cfg.value_heads * length, 0.0);
    linear_into(
        &weights.a,
        cfg.hidden,
        cfg.value_heads,
        &ws.masked,
        length,
        &mut ws.a_proj,
    );

    ws.b_proj.resize(cfg.value_heads * length, 0.0);
    linear_into(
        &weights.b,
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

    ws.q.resize(kd * length, 0.0);
    ws.k.resize(kd * length, 0.0);
    ws.v.resize(vd * length, 0.0);
    ws.z.resize(vd * length, 0.0);
    ws.scratch.resize(kd.max(vd), 0.0);
    ws.out.resize(value_width * length, 0.0);

    let inv_scale = 1.0 / (kd as f32).sqrt();
    for head in 0..cfg.value_heads {
        let key_head = cfg.key_head_for(head);
        let q_offset = key_head * kd;
        let k_offset = key_width + key_head * kd;
        let v_offset = 2 * key_width + head * vd;
        let z_offset = head * vd;

        for row in 0..kd {
            for t in 0..length {
                ws.q[row * length + t] = ws.mixed[(q_offset + row) * length + t];
                ws.k[row * length + t] = ws.mixed[(k_offset + row) * length + t];
            }
        }
        for row in 0..vd {
            for t in 0..length {
                ws.v[row * length + t] = ws.mixed[(v_offset + row) * length + t];
                ws.z[row * length + t] = ws.z_proj[(z_offset + row) * length + t];
            }
        }

        // L2-normalize Q and K per position; Q is additionally scaled by
        // `1 / sqrt(key_dim)` (see the layer reference for the orientation note).
        for t in 0..length {
            for d in 0..kd {
                ws.scratch[d] = ws.q[d * length + t];
            }
            l2_normalize(&mut ws.scratch[..kd], 1e-6);
            for d in 0..kd {
                ws.q[d * length + t] = ws.scratch[d] * inv_scale;
            }
            for d in 0..kd {
                ws.scratch[d] = ws.k[d * length + t];
            }
            l2_normalize(&mut ws.scratch[..kd], 1e-6);
            for d in 0..kd {
                ws.k[d * length + t] = ws.scratch[d];
            }
        }

        ws.state.resize(vd * kd, 0.0);
        ws.prediction.resize(vd, 0.0);
        ws.correction.resize(vd, 0.0);
        ws.result.resize(vd, 0.0);
        ws.state.iter_mut().for_each(|value| *value = 0.0);

        for t in 0..length {
            let factor = ws.decay[head * length + t].exp();
            let beta = ws.beta[head * length + t];

            for v in 0..vd {
                let mut acc = 0.0f32;
                for d in 0..kd {
                    acc += ws.state[v * kd + d] * ws.k[d * length + t];
                }
                ws.prediction[v] = acc;
            }
            for v in 0..vd {
                ws.correction[v] = beta * (ws.v[v * length + t] - factor * ws.prediction[v]);
            }
            for v in 0..vd {
                for d in 0..kd {
                    ws.state[v * kd + d] =
                        factor * ws.state[v * kd + d] + ws.correction[v] * ws.k[d * length + t];
                }
            }
            for v in 0..vd {
                let mut acc = 0.0f32;
                for d in 0..kd {
                    acc += ws.state[v * kd + d] * ws.q[d * length + t];
                }
                ws.result[v] = acc;
            }

            rms_noncentered_in_place(&mut ws.result, &weights.norm, cfg.eps);
            for v in 0..vd {
                ws.out[(head * vd + v) * length + t] = ws.result[v] * silu(ws.z[v * length + t]);
            }
        }
    }

    ws.output.resize(cfg.hidden * length, 0.0);
    linear_into(
        &weights.out_proj,
        value_width,
        cfg.hidden,
        &ws.out,
        length,
        &mut ws.output,
    );
    Ok(&ws.output)
}
