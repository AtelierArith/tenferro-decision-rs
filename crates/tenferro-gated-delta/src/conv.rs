//! Causal depthwise convolution with a fused SiLU.
//!
//! Layout is row-major `(channels, length)`: index `channel * length + t`.
//! Weights are `(taps, channels)`: index `tap * channels + channel`.

use crate::ops::silu;

/// `out[ch, t] = silu(sum_tap in[ch, t - (taps-1-tap)] * w[tap, ch])`.
pub fn causal_depthwise_silu(
    input: &[f32],
    channels: usize,
    length: usize,
    weight: &[f32],
    taps: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; channels * length];
    causal_depthwise_silu_into(input, channels, length, weight, taps, &mut output);
    output
}

/// The fused kernel behind [`causal_depthwise_silu`], writing into `output`.
///
/// `output` must have `channels * length` elements.
pub fn causal_depthwise_silu_into(
    input: &[f32],
    channels: usize,
    length: usize,
    weight: &[f32],
    taps: usize,
    output: &mut [f32],
) {
    for channel in 0..channels {
        let row = channel * length;
        for t in 0..length {
            let mut acc = 0.0f32;
            for tap in 0..taps {
                let lag = taps - 1 - tap;
                if t >= lag {
                    acc += input[row + (t - lag)] * weight[tap * channels + channel];
                }
            }
            output[row + t] = silu(acc);
        }
    }
}
