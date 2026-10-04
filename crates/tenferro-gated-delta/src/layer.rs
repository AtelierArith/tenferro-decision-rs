//! Full Gated DeltaNet layer: projections, causal convolution, Q/K
//! normalization, the scan, and the output projection.
//!
//! Two implementations are provided so they can be cross-checked:
//!
//! - [`delta_layer_reference`]: all host, using the recurrent scan.
//! - [`delta_layer_tenferro`]: projections and the chunked scan run through the
//!   eager session; the causal convolution and normalization are host helpers
//!   for now (documented optimization target).
//!
//! Weight layouts are row-major `(in, out)`, matching `native_linear(W, x) =
//! W^T x` in the Julia reference.

use decision_core::DecisionError;
use tenferro_ad::{DotGeneralConfig, EagerSession, EagerTensor, Result as AdResult};

use crate::chunked::{col_major, constant, delta_scan_chunked};
use crate::config::GatedDeltaConfig;
use crate::conv::causal_depthwise_silu;
use crate::ops::{l2_normalize, sigmoid, softplus};
use crate::reference::{delta_scan_reference, DeltaScanInputs};

/// Prepared layer weights, row-major `(in, out)`.
#[derive(Clone, Debug)]
pub struct GatedDeltaWeights {
    /// Fused q/k/v projection `(hidden, 2*key_width + value_width)`.
    pub qkv: Vec<f32>,
    /// Output gate projection `(hidden, value_width)`.
    pub z: Vec<f32>,
    /// Decay projection `(hidden, value_heads)`.
    pub a: Vec<f32>,
    /// Write-strength projection `(hidden, value_heads)`.
    pub b: Vec<f32>,
    /// Depthwise convolution kernel `(conv_taps, conv_channels)`.
    pub conv: Vec<f32>,
    /// Precomputed `-exp(A_log)` `(value_heads,)`.
    pub a_decay: Vec<f32>,
    /// Decay bias `(value_heads,)`.
    pub dt_bias: Vec<f32>,
    /// Non-centered RMSNorm scale `(value_dim,)`.
    pub norm: Vec<f32>,
    /// Output projection `(value_width, hidden)`.
    pub out_proj: Vec<f32>,
}

impl GatedDeltaWeights {
    /// Validate tensor lengths against the config.
    pub fn validate(&self, cfg: &GatedDeltaConfig) -> Result<(), DecisionError> {
        let key_width = cfg.key_dim * cfg.key_heads;
        let value_width = cfg.value_dim * cfg.value_heads;
        let conv_channels = 2 * key_width + value_width;
        let expected = [
            ("qkv", self.qkv.len(), cfg.hidden * conv_channels),
            ("z", self.z.len(), cfg.hidden * value_width),
            ("a", self.a.len(), cfg.hidden * cfg.value_heads),
            ("b", self.b.len(), cfg.hidden * cfg.value_heads),
            ("conv", self.conv.len(), cfg.conv_taps * conv_channels),
            ("a_decay", self.a_decay.len(), cfg.value_heads),
            ("dt_bias", self.dt_bias.len(), cfg.value_heads),
            ("norm", self.norm.len(), cfg.value_dim),
            ("out_proj", self.out_proj.len(), value_width * cfg.hidden),
        ];
        for (name, found, want) in expected {
            if found != want {
                return Err(DecisionError::invalid_field(
                    format!("gated_delta.{name}"),
                    format!("expected {want} values, found {found}"),
                ));
            }
        }
        Ok(())
    }
}

fn linear_host(
    weight: &[f32],
    in_dim: usize,
    out_dim: usize,
    x: &[f32],
    length: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim * length];
    for o in 0..out_dim {
        for t in 0..length {
            let mut acc = 0.0f32;
            for i in 0..in_dim {
                acc += weight[i * out_dim + o] * x[i * length + t];
            }
            y[o * length + t] = acc;
        }
    }
    y
}

fn mask_rows(x: &[f32], hidden: usize, length: usize, mask: &[f32]) -> Vec<f32> {
    let mut masked = x.to_vec();
    for i in 0..hidden {
        for t in 0..length {
            masked[i * length + t] *= mask[t];
        }
    }
    masked
}

struct OwnedInputs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    z: Vec<f32>,
    beta: Vec<f32>,
    decay: Vec<f32>,
}

fn build_head_inputs(
    cfg: &GatedDeltaConfig,
    mixed: &[f32],
    z_rows: &[f32],
    beta: &[f32],
    decay: &[f32],
    length: usize,
    value_head: usize,
) -> OwnedInputs {
    let kd = cfg.key_dim;
    let vd = cfg.value_dim;
    let key_width = kd * cfg.key_heads;
    let key_head = cfg.key_head_for(value_head);

    let q_offset = key_head * kd;
    let k_offset = key_width + key_head * kd;
    let v_offset = 2 * key_width + value_head * vd;
    let z_offset = value_head * vd;

    let mut q = vec![0.0f32; kd * length];
    let mut k = vec![0.0f32; kd * length];
    for row in 0..kd {
        for t in 0..length {
            q[row * length + t] = mixed[(q_offset + row) * length + t];
            k[row * length + t] = mixed[(k_offset + row) * length + t];
        }
    }
    // The reference L2-normalizes `q` and scales it by `1 / sqrt(key_dim)`
    // (`query ./= sqrt(sum + eps) .* sqrt(key_dim)` in the Julia reference and
    // HF Qwen3.5), and L2-normalizes `k`. Keep the same orientation so the
    // epsilon in the output RMSNorm matches the reference bit for bit.
    let inv_scale = 1.0 / (kd as f32).sqrt();
    for t in 0..length {
        let mut qcol: Vec<f32> = (0..kd).map(|d| q[d * length + t]).collect();
        l2_normalize(&mut qcol, 1e-6);
        let mut kcol: Vec<f32> = (0..kd).map(|d| k[d * length + t]).collect();
        l2_normalize(&mut kcol, 1e-6);
        for d in 0..kd {
            q[d * length + t] = qcol[d] * inv_scale;
            k[d * length + t] = kcol[d];
        }
    }

    let v: Vec<f32> = (0..vd * length)
        .map(|index| mixed[v_offset * length + index])
        .collect();
    let z: Vec<f32> = (0..vd * length)
        .map(|index| z_rows[z_offset * length + index])
        .collect();
    let beta: Vec<f32> = beta[value_head * length..(value_head + 1) * length].to_vec();
    let decay: Vec<f32> = decay[value_head * length..(value_head + 1) * length].to_vec();

    OwnedInputs {
        q,
        k,
        v,
        z,
        beta,
        decay,
    }
}

/// Compute the `beta` and `decay` rows (value_heads, length) from the
/// projections.
fn gates(
    cfg: &GatedDeltaConfig,
    weights: &GatedDeltaWeights,
    a_proj: &[f32],
    b_proj: &[f32],
    length: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut beta = vec![0.0f32; cfg.value_heads * length];
    let mut decay = vec![0.0f32; cfg.value_heads * length];
    for head in 0..cfg.value_heads {
        for t in 0..length {
            let index = head * length + t;
            beta[index] = sigmoid(b_proj[index]);
            decay[index] = weights.a_decay[head] * softplus(a_proj[index] + weights.dt_bias[head]);
        }
    }
    (beta, decay)
}

/// Full host reference layer. Returns `(hidden, length)` row-major.
pub fn delta_layer_reference(
    cfg: &GatedDeltaConfig,
    weights: &GatedDeltaWeights,
    x: &[f32],
    mask: &[f32],
) -> decision_core::Result<Vec<f32>> {
    weights.validate(cfg)?;
    let length = mask.len();
    let key_width = cfg.key_dim * cfg.key_heads;
    let value_width = cfg.value_dim * cfg.value_heads;
    let conv_channels = 2 * key_width + value_width;

    let masked = mask_rows(x, cfg.hidden, length, mask);
    let qkv_proj = linear_host(&weights.qkv, cfg.hidden, conv_channels, &masked, length);
    let mixed = causal_depthwise_silu(
        &qkv_proj,
        conv_channels,
        length,
        &weights.conv,
        cfg.conv_taps,
    );
    let z_proj = linear_host(&weights.z, cfg.hidden, value_width, &masked, length);
    let a_proj = linear_host(&weights.a, cfg.hidden, cfg.value_heads, &masked, length);
    let b_proj = linear_host(&weights.b, cfg.hidden, cfg.value_heads, &masked, length);
    let (beta, decay) = gates(cfg, weights, &a_proj, &b_proj, length);

    let mut out = vec![0.0f32; value_width * length];
    for head in 0..cfg.value_heads {
        let owned = build_head_inputs(cfg, &mixed, &z_proj, &beta, &decay, length, head);
        let inputs = DeltaScanInputs {
            q: &owned.q,
            k: &owned.k,
            v: &owned.v,
            beta: &owned.beta,
            decay: &owned.decay,
            z: &owned.z,
            norm: &weights.norm,
            eps: cfg.eps,
            key_dim: cfg.key_dim,
            value_dim: cfg.value_dim,
            length,
        };
        let head_out = delta_scan_reference(&inputs);
        let offset = head * cfg.value_dim * length;
        out[offset..offset + cfg.value_dim * length].copy_from_slice(&head_out);
    }

    Ok(linear_host(
        &weights.out_proj,
        value_width,
        cfg.hidden,
        &out,
        length,
    ))
}

fn invalid_weights(error: DecisionError) -> tenferro_ad::Error {
    tenferro_ad::Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
        "tenferro-gated-delta",
        "weights",
        error.to_string(),
    ))
}

fn linear_col(
    session: &mut EagerSession<'_>,
    x: &EagerTensor,
    weight: &EagerTensor,
) -> AdResult<EagerTensor> {
    // x [in, L], weight [in, out] -> [out, L]
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

fn extract_row_major(
    session: &mut EagerSession<'_>,
    tensor: &EagerTensor,
    rows: usize,
    cols: usize,
) -> AdResult<Vec<f32>> {
    let host = session.duplicate_value(tensor)?;
    let values = host.as_slice::<f32>()?;
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            out[row * cols + col] = values[row + col * rows];
        }
    }
    Ok(out)
}

/// Full layer with tenferro-backed projections and chunked scan.
///
/// Returns a tenferro result because it runs inside an eager session.
pub fn delta_layer_tenferro(
    session: &mut EagerSession<'_>,
    cfg: &GatedDeltaConfig,
    weights: &GatedDeltaWeights,
    x: &[f32],
    mask: &[f32],
) -> tenferro_ad::Result<Vec<f32>> {
    weights.validate(cfg).map_err(invalid_weights)?;
    let length = mask.len();
    let key_width = cfg.key_dim * cfg.key_heads;
    let value_width = cfg.value_dim * cfg.value_heads;
    let conv_channels = 2 * key_width + value_width;

    let masked = mask_rows(x, cfg.hidden, length, mask);
    let x_t = constant(
        session,
        &[cfg.hidden, length],
        &col_major(cfg.hidden, length, &masked)?,
    )?;

    let qkv_w = constant(
        session,
        &[cfg.hidden, conv_channels],
        &col_major(cfg.hidden, conv_channels, &weights.qkv)?,
    )?;
    let qkv_t = linear_col(session, &x_t, &qkv_w)?;
    let qkv_proj = extract_row_major(session, &qkv_t, conv_channels, length)?;
    let mixed = causal_depthwise_silu(
        &qkv_proj,
        conv_channels,
        length,
        &weights.conv,
        cfg.conv_taps,
    );

    let z_w = constant(
        session,
        &[cfg.hidden, value_width],
        &col_major(cfg.hidden, value_width, &weights.z)?,
    )?;
    let z_t = linear_col(session, &x_t, &z_w)?;
    let z_proj = extract_row_major(session, &z_t, value_width, length)?;

    let a_w = constant(
        session,
        &[cfg.hidden, cfg.value_heads],
        &col_major(cfg.hidden, cfg.value_heads, &weights.a)?,
    )?;
    let b_w = constant(
        session,
        &[cfg.hidden, cfg.value_heads],
        &col_major(cfg.hidden, cfg.value_heads, &weights.b)?,
    )?;
    let a_t = linear_col(session, &x_t, &a_w)?;
    let b_t = linear_col(session, &x_t, &b_w)?;
    let a_proj = extract_row_major(session, &a_t, cfg.value_heads, length)?;
    let b_proj = extract_row_major(session, &b_t, cfg.value_heads, length)?;
    let (beta, decay) = gates(cfg, weights, &a_proj, &b_proj, length);

    let mut out = vec![0.0f32; value_width * length];
    for head in 0..cfg.value_heads {
        let owned = build_head_inputs(cfg, &mixed, &z_proj, &beta, &decay, length, head);
        let inputs = DeltaScanInputs {
            q: &owned.q,
            k: &owned.k,
            v: &owned.v,
            beta: &owned.beta,
            decay: &owned.decay,
            z: &owned.z,
            norm: &weights.norm,
            eps: cfg.eps,
            key_dim: cfg.key_dim,
            value_dim: cfg.value_dim,
            length,
        };
        let head_out = delta_scan_chunked(session, &inputs, cfg.chunk_size)?;
        let offset = head * cfg.value_dim * length;
        out[offset..offset + cfg.value_dim * length].copy_from_slice(&head_out);
    }

    let out_t = constant(
        session,
        &[value_width, length],
        &col_major(value_width, length, &out)?,
    )?;
    let out_w = constant(
        session,
        &[value_width, cfg.hidden],
        &col_major(value_width, cfg.hidden, &weights.out_proj)?,
    )?;
    let final_t = linear_col(session, &out_t, &out_w)?;
    extract_row_major(session, &final_t, cfg.hidden, length)
}
