//! Host scalar/vector helpers matching the Julia reference definitions.

/// Logistic sigmoid.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// SiLU / swish.
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// `softplus(x) = max(x, 0) + log1p(exp(-|x|))`.
pub fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// In-place L2 normalization with the given epsilon.
pub fn l2_normalize(values: &mut [f32], eps: f32) {
    let sum_sq: f32 = values.iter().map(|v| v * v).sum();
    let inv = 1.0 / (sum_sq + eps).sqrt();
    for value in values.iter_mut() {
        *value *= inv;
    }
}

/// Non-centered RMSNorm of a `value_dim` vector: `x * rsqrt(mean(x^2)+eps) * w`.
pub fn rms_noncentered(values: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let d = values.len() as f32;
    let mean_sq: f32 = values.iter().map(|v| v * v).sum::<f32>() / d;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    values
        .iter()
        .zip(weight)
        .map(|(x, w)| x * inv * w)
        .collect()
}
