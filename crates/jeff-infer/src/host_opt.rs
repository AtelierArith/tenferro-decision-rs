//! Host-optimized Jeff forward (`forward_host_opt`).
//!
//! This is the same math as [`crate::model::forward_reference`] — the
//! correctness oracle, which is left untouched — but structured to close the
//! CPU gap to the Julia runtime (`docs/agents/specs/docs/21_SPEED_COMPARISON.md`)
//! by porting the techniques from `extern/JeffClient.jl`'s native CPU runtime:
//!
//! - **rayon parallelism at the right granularity**: the oracle parallelizes
//!   only inside GEMMs and the DeltaNet head loop, leaving RMSNorm, RoPE, the
//!   MLP SiLU gate, the attention head loop, and the residual adds
//!   single-threaded. Here the elementwise work is spread over tokens, heads,
//!   and elements.
//! - **RoPE tables precomputed once per length**, instead of recomputing
//!   `theta.powf(...)` and `cos`/`sin` for every `(head, token, channel)`.
//! - **A [`HostOptWorkspace`] that reuses every activation buffer** across the
//!   layer stack, so a warmed forward is allocation-free. The oracle allocates a
//!   fresh `Vec` for every intermediate (and copies the DeltaNet output).
//!
//! The DeltaNet layer reuses the fused host recurrent kernel
//! ([`delta_layer_recurrent`]), which already parallelizes the value heads and
//! owns its scratch buffers. The oracle stays the reference this path is tested
//! against (`crates/jeff-infer/tests/model.rs`).

use rayon::prelude::*;

use decision_core::{DecisionError, Result};
use tenferro_gated_delta::{GatedDeltaWorkspace, delta_layer_recurrent};

use crate::model::{AttentionWeights, FullAttentionWeights, JeffConfig, JeffWeights, MlpWeights};

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Centered RMSNorm over the rows of a `(hidden, length)` row-major activation.
///
/// `means[t]` receives the per-token `1/sqrt(mean(x²)+eps)`; the second pass is
/// parallel over rows.
fn rms_centered_into(
    x: &[f32],
    weight: &[f32],
    hidden: usize,
    length: usize,
    eps: f32,
    means: &mut [f32],
    y: &mut [f32],
) {
    debug_assert_eq!(y.len(), hidden * length);
    let inv_hidden = 1.0 / hidden as f32;
    means[..length]
        .par_iter_mut()
        .enumerate()
        .for_each(|(t, mean)| {
            let mut acc = 0.0f32;
            for d in 0..hidden {
                let v = x[d * length + t];
                acc += v * v;
            }
            *mean = 1.0 / (acc * inv_hidden + eps).sqrt();
        });
    y.par_chunks_mut(length).enumerate().for_each(|(d, row)| {
        let scale = 1.0 + weight[d];
        let base = d * length;
        for (t, value) in row.iter_mut().enumerate() {
            *value = x[base + t] * means[t] * scale;
        }
    });
}

/// RMSNorm over the head dimension of a `(heads*head_dim, length)` activation,
/// parallel over heads (each head owns disjoint rows).
fn rms_heads_into(
    x: &[f32],
    head_dim: usize,
    length: usize,
    weight: &[f32],
    eps: f32,
    y: &mut [f32],
) {
    debug_assert_eq!(y.len(), x.len());
    debug_assert_eq!(y.len() % (head_dim * length), 0);
    y.par_chunks_mut(head_dim * length)
        .enumerate()
        .for_each(|(head, rows)| {
            let base = head * head_dim;
            for t in 0..length {
                let mut acc = 0.0f32;
                for d in 0..head_dim {
                    let v = x[(base + d) * length + t];
                    acc += v * v;
                }
                let scale = 1.0 / (acc / head_dim as f32 + eps).sqrt();
                for d in 0..head_dim {
                    rows[d * length + t] = x[(base + d) * length + t] * scale * (1.0 + weight[d]);
                }
            }
        });
}

/// In-place partial RoPE on a `(heads*head_dim, length)` activation, parallel
/// over heads. `cos`/`sin` are `(rotary_dim/2, length)` tables.
fn rope_into(x: &mut [f32], head_dim: usize, length: usize, half: usize, cos: &[f32], sin: &[f32]) {
    debug_assert_eq!(x.len() % (head_dim * length), 0);
    x.par_chunks_mut(head_dim * length).for_each(|rows| {
        for t in 0..length {
            for i in 0..half {
                let a = rows[i * length + t];
                let b = rows[(i + half) * length + t];
                let c = cos[i * length + t];
                let s = sin[i * length + t];
                rows[i * length + t] = a * c - b * s;
                rows[(i + half) * length + t] = a * s + b * c;
            }
        }
    });
}

/// Full attention on a `(hidden, length)` input, writing the `(hidden, length)`
/// projection of the gated attention output into `out`.
///
/// The oracle runs the head loop serially with a shared `scores` scratch; here
/// each head runs in parallel with its own scratch. The math is identical.
#[allow(clippy::too_many_arguments)]
fn full_attention_into(
    w: &FullAttentionWeights,
    cfg: &JeffConfig,
    x: &[f32],
    mask: &[f32],
    q: &mut [f32],
    q_gate: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    q_norm: &mut [f32],
    k_norm: &mut [f32],
    attn_out: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    out: &mut [f32],
) {
    let length = mask.len();
    let hd = cfg.head_dim;
    let heads = cfg.heads;
    let width = hd * heads;
    let half = w.rotary_dim / 2;

    cpu_kernels::matmul_row_major_into(&w.q, cfg.hidden, width, x, length, q);
    cpu_kernels::matmul_row_major_into(&w.gate, cfg.hidden, width, x, length, q_gate);
    cpu_kernels::matmul_row_major_into(&w.k, cfg.hidden, width, x, length, k);
    cpu_kernels::matmul_row_major_into(&w.v, cfg.hidden, width, x, length, v);

    rms_heads_into(q, hd, length, &w.q_norm, cfg.eps, q_norm);
    rms_heads_into(k, hd, length, &w.k_norm, cfg.eps, k_norm);
    rope_into(q_norm, hd, length, half, cos, sin);
    rope_into(k_norm, hd, length, half, cos, sin);

    let scale = 1.0 / (hd as f32).sqrt();
    attn_out
        .par_chunks_mut(hd * length)
        .enumerate()
        .for_each_init(
            || vec![0.0f32; length],
            |scores, (head, out_rows)| {
                let base = head * hd;
                for query in 0..length {
                    let mut max = f32::NEG_INFINITY;
                    for key in 0..length {
                        let value = if key <= query && mask[key] != 0.0 {
                            let mut acc = 0.0f32;
                            for d in 0..hd {
                                acc += k_norm[(base + d) * length + key]
                                    * q_norm[(base + d) * length + query];
                            }
                            acc * scale
                        } else {
                            f32::NEG_INFINITY
                        };
                        scores[key] = value;
                        if value > max {
                            max = value;
                        }
                    }
                    let mut sum = 0.0f32;
                    for score in scores.iter_mut() {
                        *score = if score.is_finite() {
                            (*score - max).exp()
                        } else {
                            0.0
                        };
                        sum += *score;
                    }
                    for d in 0..hd {
                        let row = base + d;
                        let mut acc = 0.0f32;
                        for key in 0..length {
                            acc += v[row * length + key] * scores[key];
                        }
                        out_rows[d * length + query] =
                            acc / sum * sigmoid(q_gate[row * length + query]);
                    }
                }
            },
        );

    cpu_kernels::matmul_row_major_into(&w.o, width, cfg.hidden, attn_out, length, out);
}

/// SiLU-gated MLP on a `(hidden, length)` input, writing `(hidden, length)`.
fn mlp_into(
    mlp: &MlpWeights,
    cfg: &JeffConfig,
    x: &[f32],
    length: usize,
    gate: &mut [f32],
    up: &mut [f32],
    out: &mut [f32],
) {
    cpu_kernels::matmul_row_major_into(&mlp.gate, cfg.hidden, cfg.intermediate, x, length, gate);
    cpu_kernels::matmul_row_major_into(&mlp.up, cfg.hidden, cfg.intermediate, x, length, up);
    gate.par_iter_mut().zip(up.par_iter()).for_each(|(g, u)| {
        let value = *g;
        *g = (value / (1.0 + (-value).exp())) * *u;
    });
    cpu_kernels::matmul_row_major_into(&mlp.down, cfg.intermediate, cfg.hidden, gate, length, out);
}

/// `out = a + b`, parallel over elements.
fn residual_add_into(a: &[f32], b: &[f32], out: &mut [f32]) {
    out.par_iter_mut()
        .zip(a.par_iter())
        .zip(b.par_iter())
        .for_each(|((o, x), y)| *o = x + y);
}

/// Gather embedding rows `(hidden, vocab)` by token id into `(hidden, length)`.
fn embedding_into(table: &[f32], vocab: usize, length: usize, ids: &[i64], out: &mut [f32]) {
    out.par_chunks_mut(length).enumerate().for_each(|(d, row)| {
        let base = d * vocab;
        for (t, id) in ids.iter().enumerate() {
            row[t] = table[base + *id as usize];
        }
    });
}

/// Reusable activation buffers for [`forward_host_opt_with`].
///
/// Buffers are sized for the last `(config, length)` seen and reused across the
/// layer stack and across calls that share a sequence length, so a warmed
/// forward performs no heap allocation.
#[derive(Clone, Debug, Default)]
pub struct HostOptWorkspace {
    hidden: usize,
    intermediate: usize,
    width: usize,
    options: usize,
    length: usize,
    configured: bool,

    state: Vec<f32>,
    normalized: Vec<f32>,
    residual: Vec<f32>,
    mixed: Vec<f32>,
    means: Vec<f32>,

    q: Vec<f32>,
    q_gate: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    attn_out: Vec<f32>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,

    mlp_gate: Vec<f32>,
    mlp_up: Vec<f32>,

    last: Vec<f32>,
    logits: Vec<f32>,

    delta: GatedDeltaWorkspace,
}

impl HostOptWorkspace {
    /// An empty workspace; buffers allocate on first use.
    pub fn new() -> Self {
        Self::default()
    }

    /// The total bytes retained by the workspace buffers.
    pub fn retained_bytes(&self) -> usize {
        [
            &self.state,
            &self.normalized,
            &self.residual,
            &self.mixed,
            &self.means,
            &self.q,
            &self.q_gate,
            &self.k,
            &self.v,
            &self.q_norm,
            &self.k_norm,
            &self.attn_out,
            &self.rope_cos,
            &self.rope_sin,
            &self.mlp_gate,
            &self.mlp_up,
            &self.last,
            &self.logits,
        ]
        .iter()
        .map(|buffer| buffer.len() * 4)
        .sum::<usize>()
            + self.delta.retained_bytes()
    }

    fn ensure(&mut self, cfg: &JeffConfig, length: usize, weights: &JeffWeights) {
        let width = cfg.heads * cfg.head_dim;
        if self.configured
            && self.length == length
            && self.hidden == cfg.hidden
            && self.intermediate == cfg.intermediate
            && self.width == width
            && self.options == weights.options
        {
            return;
        }
        self.hidden = cfg.hidden;
        self.intermediate = cfg.intermediate;
        self.width = width;
        self.options = weights.options;
        self.length = length;
        self.configured = true;

        let hl = cfg.hidden * length;
        let wl = width * length;
        let il = cfg.intermediate * length;
        self.state = vec![0.0; hl];
        self.normalized = vec![0.0; hl];
        self.residual = vec![0.0; hl];
        self.mixed = vec![0.0; hl];
        self.means = vec![0.0; length];
        self.q = vec![0.0; wl];
        self.q_gate = vec![0.0; wl];
        self.k = vec![0.0; wl];
        self.v = vec![0.0; wl];
        self.q_norm = vec![0.0; wl];
        self.k_norm = vec![0.0; wl];
        self.attn_out = vec![0.0; wl];
        self.mlp_gate = vec![0.0; il];
        self.mlp_up = vec![0.0; il];
        self.last = vec![0.0; cfg.hidden];
        self.logits = vec![0.0; weights.options];

        // RoPE tables: `(rotary_dim/2, length)`, shared by every full-attention
        // layer. Built from the first full-attention layer's base.
        let rotary = weights
            .layers
            .iter()
            .find_map(|layer| match &layer.attention {
                AttentionWeights::Full(w) => Some((w.rotary_dim, w.rope_theta)),
                AttentionWeights::Delta { .. } => None,
            });
        match rotary {
            Some((rotary_dim, theta)) if rotary_dim > 0 => {
                let half = rotary_dim / 2;
                self.rope_cos = vec![0.0; half * length];
                self.rope_sin = vec![0.0; half * length];
                for i in 0..half {
                    let freq = theta.powf(-2.0 * i as f32 / rotary_dim as f32);
                    for t in 0..length {
                        let angle = t as f32 * freq;
                        self.rope_cos[i * length + t] = angle.cos();
                        self.rope_sin[i * length + t] = angle.sin();
                    }
                }
            }
            _ => {
                self.rope_cos.clear();
                self.rope_sin.clear();
            }
        }
    }
}

/// Host-optimized forward. Returns readout scores `(options,)`.
///
/// Numerically equivalent to [`crate::model::forward_reference`] (the oracle)
/// but parallel and allocation-free after warmup.
pub fn forward_host_opt(
    cfg: &JeffConfig,
    weights: &JeffWeights,
    ids: &[i64],
    mask: &[f32],
) -> Result<Vec<f32>> {
    let mut workspace = HostOptWorkspace::new();
    forward_host_opt_with(&mut workspace, cfg, weights, ids, mask)
}

/// [`forward_host_opt`] with a reusable [`HostOptWorkspace`].
pub fn forward_host_opt_with(
    ws: &mut HostOptWorkspace,
    cfg: &JeffConfig,
    weights: &JeffWeights,
    ids: &[i64],
    mask: &[f32],
) -> Result<Vec<f32>> {
    weights.validate(cfg)?;
    let length = ids.len();
    if mask.len() != length {
        return Err(DecisionError::invalid_field(
            "jeff.mask",
            "mask length must match the token length",
        ));
    }
    if ids
        .iter()
        .any(|id| *id < 0 || *id as usize >= weights.vocab)
    {
        return Err(DecisionError::invalid_field(
            "jeff.ids",
            "token id is outside the vocabulary",
        ));
    }
    ws.ensure(cfg, length, weights);

    embedding_into(
        &weights.embedding,
        weights.vocab,
        length,
        ids,
        &mut ws.state,
    );

    for layer in &weights.layers {
        rms_centered_into(
            &ws.state,
            &layer.input_norm,
            cfg.hidden,
            length,
            cfg.eps,
            &mut ws.means,
            &mut ws.normalized,
        );

        match &layer.attention {
            AttentionWeights::Full(w) => full_attention_into(
                w,
                cfg,
                &ws.normalized,
                mask,
                &mut ws.q,
                &mut ws.q_gate,
                &mut ws.k,
                &mut ws.v,
                &mut ws.q_norm,
                &mut ws.k_norm,
                &mut ws.attn_out,
                &ws.rope_cos,
                &ws.rope_sin,
                &mut ws.mixed,
            ),
            AttentionWeights::Delta { weights, config } => {
                let mixed =
                    delta_layer_recurrent(config, weights, &ws.normalized, mask, &mut ws.delta)?;
                ws.mixed.copy_from_slice(mixed);
            }
        }

        residual_add_into(&ws.state, &ws.mixed, &mut ws.residual);

        rms_centered_into(
            &ws.residual,
            &layer.post_norm,
            cfg.hidden,
            length,
            cfg.eps,
            &mut ws.means,
            &mut ws.normalized,
        );
        mlp_into(
            &layer.mlp,
            cfg,
            &ws.normalized,
            length,
            &mut ws.mlp_gate,
            &mut ws.mlp_up,
            &mut ws.mixed,
        );

        residual_add_into(&ws.residual, &ws.mixed, &mut ws.state);
    }

    // Final centered RMSNorm on the last position, then the readout.
    {
        let state = &ws.state;
        let last = &mut ws.last;
        last.par_iter_mut().enumerate().for_each(|(d, value)| {
            *value = state[d * length + (length - 1)];
        });
        let mean_sq: f32 = last.par_iter().map(|v| v * v).sum::<f32>() / cfg.hidden as f32;
        let scale = 1.0 / (mean_sq + cfg.eps).sqrt();
        last.par_iter_mut().enumerate().for_each(|(d, value)| {
            *value = *value * scale * (1.0 + weights.final_norm[d]);
        });
    }
    cpu_kernels::matmul_row_major_into(
        &weights.readout,
        cfg.hidden,
        weights.options,
        &ws.last,
        1,
        &mut ws.logits,
    );

    Ok(ws.logits.clone())
}
