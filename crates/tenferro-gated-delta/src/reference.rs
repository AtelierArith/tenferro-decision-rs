//! Host recurrent Gated DeltaNet reference (one value head at a time).
//!
//! Follows `12_TENFERRO_GATED_DELTA.md` §6 and the Julia reference. Inputs are
//! already prepared: `q` is L2-normalized and scaled by `1 / sqrt(key_dim)`,
//! `k` is L2-normalized. Layouts are row-major `(d, length)`.

use crate::ops::{rms_noncentered, silu};

/// Prepared per-value-head scan inputs.
pub struct DeltaScanInputs<'a> {
    /// Query/head width (`key_dim * length` values).
    pub q: &'a [f32],
    /// Key width (`key_dim * length` values).
    pub k: &'a [f32],
    /// Value width (`value_dim * length` values).
    pub v: &'a [f32],
    /// Write strength per position (`length` values).
    pub beta: &'a [f32],
    /// Log-decay per position (`length` values).
    pub decay: &'a [f32],
    /// Output gate (`value_dim * length` values).
    pub z: &'a [f32],
    /// RMSNorm scale (`value_dim` values).
    pub norm: &'a [f32],
    /// RMSNorm epsilon.
    pub eps: f32,
    /// Key/query width.
    pub key_dim: usize,
    /// Value width.
    pub value_dim: usize,
    /// Sequence length.
    pub length: usize,
}

/// Run the recurrent scan and return the head output of shape
/// `(value_dim, length)`, row-major.
pub fn delta_scan_reference(inputs: &DeltaScanInputs<'_>) -> Vec<f32> {
    let kd = inputs.key_dim;
    let vd = inputs.value_dim;
    let length = inputs.length;

    let mut state = vec![0.0f32; vd * kd]; // (value_dim, key_dim), row-major
    let mut out = vec![0.0f32; vd * length];

    let mut prediction = vec![0.0f32; vd];
    let mut correction = vec![0.0f32; vd];
    let mut result = vec![0.0f32; vd];

    for t in 0..length {
        let factor = inputs.decay[t].exp();
        let beta = inputs.beta[t];

        for (v, pred) in prediction.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for d in 0..kd {
                acc += state[v * kd + d] * inputs.k[d * length + t];
            }
            *pred = acc;
        }
        for v in 0..vd {
            correction[v] = beta * (inputs.v[v * length + t] - factor * prediction[v]);
        }
        for v in 0..vd {
            for d in 0..kd {
                state[v * kd + d] =
                    factor * state[v * kd + d] + correction[v] * inputs.k[d * length + t];
            }
        }
        for v in 0..vd {
            let mut acc = 0.0f32;
            for d in 0..kd {
                acc += state[v * kd + d] * inputs.q[d * length + t];
            }
            result[v] = acc;
        }

        let normalized = rms_noncentered(&result, inputs.norm, inputs.eps);
        for v in 0..vd {
            out[v * length + t] = normalized[v] * silu(inputs.z[v * length + t]);
        }
    }
    out
}
