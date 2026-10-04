//! Laya ModernBERT encoder and decision-head forward pass.
//!
//! This module mirrors `extern/Laya.jl/src/{model,layers,mathfns}.jl` and the
//! `laya_mlx/model.py` it follows. Two forwards are provided:
//!
//! - [`forward_encoder_reference`] / [`forward_reference`]: all-host references
//!   that are the parity baseline.
//! - [`forward_encoder_tenferro`] / [`forward_tenferro`]: the same computation
//!   through the eager tenferro session and the `tenferro-infer` primitives.
//!
//! ## Layout and weight conventions
//!
//! Everything follows Julia's column-major axes, reversed from NumPy/MLX:
//!
//! - activations are `(d, L, B)`, flattened as `index = d_index + d * (l + L * b)`
//! - [`LinearWeights::weight`] is `(in, out)`, flattened as `weight[i + in * o]`;
//!   a linear applies `y = Wᵀ x`
//! - [`LayerNormWeights::weight`] is `(d,)` and normalizes the feature axis
//!   (axis 0); the multiplicative weight is applied directly (not `1 + weight`)
//! - `tok_embeddings` is `(hidden, vocab)` and `type_emb` is `(hidden, 3)`
//!
//! Token ids, marker positions and `qtype` are 0-based, as on the Python
//! boundary.
//!
//! ## GELU
//!
//! The Laya reference uses the erf-based exact GELU (`mlx_erf`). tenferro has
//! no `erf` op, so the parity path (both the host reference and the tenferro
//! forward) uses the **tanh approximation** and is compared against itself.
//! [`exact_gelu`] / [`erf`] mirror `mathfns.jl` for future checkpoint parity
//! once a tenferro `erf` extension op lands (tracked as issue #1973).

use decision_core::{DecisionError, Result};
use tenferro_ad::{DotGeneralConfig, EagerSession, EagerTensor, GatherConfig, SliceConfig, Tensor};
use tenferro_infer::{activation, attention, norm, rope};

use crate::config::{AgentConfig, EncoderConfig, LayerKind};

// ------------------------------------------------------------------- weights

/// A LayerNorm over the feature axis: `y = (x - μ) / sqrt(σ² + eps) * weight + bias`.
#[derive(Clone, Debug)]
pub struct LayerNormWeights {
    /// Multiplicative scale `(d,)`.
    pub weight: Vec<f32>,
    /// Optional additive bias `(d,)`.
    pub bias: Option<Vec<f32>>,
}

/// A dense layer with weight `(in, out)` and optional bias `(out,)`.
#[derive(Clone, Debug)]
pub struct LinearWeights {
    /// Column-major `(in, out)` weight; `y = Wᵀ x`.
    pub weight: Vec<f32>,
    /// Optional additive bias `(out,)`.
    pub bias: Option<Vec<f32>>,
}

/// One ModernBERT encoder layer.
#[derive(Clone, Debug)]
pub struct EncoderLayerWeights {
    /// Full or sliding attention.
    pub kind: LayerKind,
    /// Input normalization; `None` for the first layer (identity).
    pub attn_norm: Option<LayerNormWeights>,
    /// Fused query/key/value projection `(d, 3d)`.
    pub wqkv: LinearWeights,
    /// Attention output projection `(d, d)`.
    pub wo: LinearWeights,
    /// Number of attention heads.
    pub num_heads: usize,
    /// RoPE base for this layer's attention kind.
    pub rope_base: f64,
    /// Post-attention (MLP) normalization `(d,)`.
    pub mlp_norm: LayerNormWeights,
    /// GeGLU up projection `(d, 2I)`.
    pub wi: LinearWeights,
    /// GeGLU down projection `(I, d)`.
    pub wo_mlp: LinearWeights,
}

/// The ModernBERT encoder weights.
#[derive(Clone, Debug)]
pub struct ModernBertWeights {
    /// Token embeddings `(hidden, vocab)`.
    pub tok_embeddings: Vec<f32>,
    /// Embedding normalization `(d,)`.
    pub embed_norm: LayerNormWeights,
    /// Encoder layers, in order.
    pub layers: Vec<EncoderLayerWeights>,
    /// Final normalization `(d,)`, applied to the last layer's output.
    pub final_norm: LayerNormWeights,
}

/// One decision-head transformer layer (pre-norm, no RoPE, ReLU MLP).
#[derive(Clone, Debug)]
pub struct HeadLayerWeights {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Pre-attention normalization `(d,)`.
    pub norm1: LayerNormWeights,
    /// Fused query/key/value projection `(d, 3d)`.
    pub in_proj: LinearWeights,
    /// Attention output projection `(d, d)`.
    pub out_proj: LinearWeights,
    /// Post-attention normalization `(d,)`.
    pub norm2: LayerNormWeights,
    /// MLP up projection `(d, ff)`.
    pub linear1: LinearWeights,
    /// MLP down projection `(ff, d)`.
    pub linear2: LinearWeights,
}

/// The full Laya decision model weights.
#[derive(Clone, Debug)]
pub struct LayaWeights {
    /// The ModernBERT encoder.
    pub encoder: ModernBertWeights,
    /// The decision-head layers.
    pub head: Vec<HeadLayerWeights>,
    /// Question-type embeddings `(hidden, 3)`.
    pub type_emb: Vec<f32>,
    /// Marker scorer normalization `(d,)`.
    pub scorer_norm: LayerNormWeights,
    /// Marker scorer first projection `(d, 1)`.
    pub scorer1: LinearWeights,
    /// Marker scorer second projection `(1, 1)`.
    pub scorer2: LinearWeights,
    /// Action-head first projection `(d + 4, A)`.
    pub act1: LinearWeights,
    /// Action-head second projection `(A, n_actions)`.
    pub act2: LinearWeights,
}

fn validate_norm(ln: &LayerNormWeights, dim: usize, field: &str) -> Result<()> {
    if ln.weight.len() != dim {
        return Err(DecisionError::invalid_field(
            field,
            format!("expected {dim} weights, found {}", ln.weight.len()),
        ));
    }
    if let Some(bias) = &ln.bias {
        if bias.len() != dim {
            return Err(DecisionError::invalid_field(
                field,
                format!("expected {dim} bias values, found {}", bias.len()),
            ));
        }
    }
    Ok(())
}

fn validate_linear(
    linear: &LinearWeights,
    in_dim: usize,
    out_dim: usize,
    field: &str,
) -> Result<()> {
    if linear.weight.len() != in_dim * out_dim {
        return Err(DecisionError::invalid_field(
            field,
            format!(
                "expected ({in_dim}, {out_dim}) = {} weights, found {}",
                in_dim * out_dim,
                linear.weight.len()
            ),
        ));
    }
    if let Some(bias) = &linear.bias {
        if bias.len() != out_dim {
            return Err(DecisionError::invalid_field(
                field,
                format!("expected {out_dim} bias values, found {}", bias.len()),
            ));
        }
    }
    Ok(())
}

impl ModernBertWeights {
    /// Validate the encoder weights against their configuration.
    pub fn validate(&self, cfg: &EncoderConfig) -> Result<()> {
        let d = cfg.hidden_size;
        let i = cfg.intermediate_size;
        if d == 0 {
            return Err(DecisionError::invalid_field(
                "encoder.hidden_size",
                "must be positive",
            ));
        }
        if self.tok_embeddings.is_empty() || self.tok_embeddings.len() % d != 0 {
            return Err(DecisionError::invalid_field(
                "laya.tok_embeddings",
                "shape must be (hidden, vocab)",
            ));
        }
        validate_norm(&self.embed_norm, d, "laya.embed_norm")?;
        validate_norm(&self.final_norm, d, "laya.final_norm")?;
        if self.layers.len() != cfg.num_hidden_layers {
            return Err(DecisionError::invalid_field(
                "laya.layers",
                format!(
                    "expected {} layers, found {}",
                    cfg.num_hidden_layers,
                    self.layers.len()
                ),
            ));
        }
        for (index, layer) in self.layers.iter().enumerate() {
            let field = format!("laya.layers[{index}]");
            if layer.num_heads == 0 || d % layer.num_heads != 0 {
                return Err(DecisionError::invalid_field(
                    format!("{field}.num_heads"),
                    "must divide hidden_size",
                ));
            }
            if (d / layer.num_heads) % 2 != 0 {
                return Err(DecisionError::invalid_field(
                    format!("{field}.num_heads"),
                    "the attention head dimension must be even for RoPE",
                ));
            }
            if let Some(attn_norm) = &layer.attn_norm {
                validate_norm(attn_norm, d, &format!("{field}.attn_norm"))?;
            }
            validate_linear(&layer.wqkv, d, 3 * d, &format!("{field}.wqkv"))?;
            validate_linear(&layer.wo, d, d, &format!("{field}.wo"))?;
            validate_norm(&layer.mlp_norm, d, &format!("{field}.mlp_norm"))?;
            validate_linear(&layer.wi, d, 2 * i, &format!("{field}.wi"))?;
            validate_linear(&layer.wo_mlp, i, d, &format!("{field}.wo_mlp"))?;
            if !(layer.rope_base.is_finite() && layer.rope_base > 0.0) {
                return Err(DecisionError::invalid_field(
                    format!("{field}.rope_base"),
                    "must be a positive finite number",
                ));
            }
        }
        Ok(())
    }
}

impl LayaWeights {
    /// Validate the full decision model weights.
    pub fn validate(&self, encoder: &EncoderConfig, agent: &AgentConfig) -> Result<()> {
        self.encoder.validate(encoder)?;
        let d = encoder.hidden_size;
        if self.type_emb.len() != d * 3 {
            return Err(DecisionError::invalid_field(
                "laya.type_emb",
                "shape must be (hidden, 3)",
            ));
        }
        validate_norm(&self.scorer_norm, d, "laya.scorer_norm")?;
        validate_linear(&self.scorer1, d, 1, "laya.scorer1")?;
        validate_linear(&self.scorer2, 1, 1, "laya.scorer2")?;
        for (index, layer) in self.head.iter().enumerate() {
            let field = format!("laya.head[{index}]");
            if layer.num_heads == 0 || d % layer.num_heads != 0 {
                return Err(DecisionError::invalid_field(
                    format!("{field}.num_heads"),
                    "must divide hidden_size",
                ));
            }
            validate_norm(&layer.norm1, d, &format!("{field}.norm1"))?;
            validate_linear(&layer.in_proj, d, 3 * d, &format!("{field}.in_proj"))?;
            validate_linear(&layer.out_proj, d, d, &format!("{field}.out_proj"))?;
            validate_norm(&layer.norm2, d, &format!("{field}.norm2"))?;
            if layer.linear1.weight.len() % d != 0 {
                return Err(DecisionError::invalid_field(
                    format!("{field}.linear1"),
                    "shape must be (hidden, ff)",
                ));
            }
            let ff = layer.linear1.weight.len() / d;
            validate_linear(&layer.linear2, ff, d, &format!("{field}.linear2"))?;
        }
        let action_hidden = self.act1.weight.len() / (d + 4);
        validate_linear(&self.act1, d + 4, action_hidden, "laya.act1")?;
        validate_linear(&self.act2, action_hidden, agent.action_count(), "laya.act2")?;
        Ok(())
    }
}

// ------------------------------------------------------------- scalar helpers

/// The tanh-approximation GELU used by the parity path.
///
/// `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`. The Laya reference
/// uses [`exact_gelu`] instead; see the module note.
pub fn gelu_tanh(x: f32) -> f32 {
    let scale = (2.0f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (scale * (x + 0.044715 * x * x * x)).tanh())
}

/// The MLX `erff` kernel, ported from `mathfns.jl` (`mlx_erf`).
pub fn erf(a: f32) -> f32 {
    mlx_erf(a)
}

/// The exact, erf-based GELU: `x * (1 + erf(x/√2)) / 2`.
///
/// This mirrors `mlx.nn.gelu` and `mathfns.jl`'s `gelu`. It is provided for
/// future checkpoint parity; the forward pass uses [`gelu_tanh`] until a
/// tenferro `erf` extension op exists (issue #1973).
pub fn exact_gelu(x: f32) -> f32 {
    x * (1.0 + erf(x / std::f32::consts::SQRT_2)) / 2.0
}

/// `expm1` as evaluated by the MLX Metal kernel (`mathfns.jl` `mlx_expm1f`).
///
/// The literal constants are the MLX Metal kernel's exactly; keep their bits.
#[allow(clippy::approx_constant, clippy::excessive_precision)]
fn mlx_expm1f(a: f32) -> f32 {
    let mut j = 1.442695f32.mul_add(a, 12582912.0);
    j -= 12582912.0;
    let i = j as i32;
    let f = j.mul_add(-0.693145_752f32, a);
    let s = if a == 0.0 { a } else { f * f };
    let mut r = 1.973_509_8e-4f32;
    r = r.mul_add(f, 1.393_090_7e-3);
    r = r.mul_add(f, 8.333_44e-3);
    r = r.mul_add(f, 4.166_680_2e-2);
    r = r.mul_add(f, 1.666_667_2e-1);
    r = r.mul_add(f, 4.999_999_7e-1);
    let u = if j == 1.0 { f + 0.5 } else { f };
    let v = r.mul_add(s, u);
    let half = 0.5f32;
    let t = half * 2.0f32.powi(i);
    let y = t - half;
    let x = (t - y) - half;
    r = v.mul_add(t, x) + y;
    r += r;
    if j == 0.0 {
        r = v;
    }
    if j == 1.0 {
        r = v + v;
    }
    if (a - 1.0).abs() > 88.0 {
        let e = a.exp2();
        r = e.mul_add(e, -1.0);
    }
    r
}

/// `erf` as evaluated by the MLX Metal kernel (`mathfns.jl` `mlx_erf`).
///
/// The literal constants are the MLX Metal kernel's exactly; keep their bits.
#[allow(clippy::excessive_precision)]
fn mlx_erf(a: f32) -> f32 {
    let t = a.abs();
    let s = a * a;
    if t > 0.927_734_4 {
        let mut r = (-1.728_534_7e-5f32).mul_add(t, 3.831_971_3e-4);
        let u = (-3.883_964_4e-3f32).mul_add(t, 2.425_462_2e-2);
        r = r.mul_add(s, u);
        r = r.mul_add(t, -1.067_778_8e-1);
        r = r.mul_add(t, -6.348_466_9e-1);
        r = r.mul_add(t, -1.287_175_1e-1);
        r = r.mul_add(t, -t);
        r = -mlx_expm1f(r);
        r.copysign(a)
    } else {
        let mut r = -5.967_617e-4f32;
        r = r.mul_add(s, 4.991_194_2e-3);
        r = r.mul_add(s, -2.676_813_5e-2);
        r = r.mul_add(s, 1.128_199_2e-1);
        r = r.mul_add(s, -3.761_253_4e-1);
        r = r.mul_add(s, 1.283_791_7e-1);
        r.mul_add(a, a)
    }
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

// ------------------------------------------------------------- host primitives

fn layer_norm_host(
    ln: &LayerNormWeights,
    x: &[f32],
    d: usize,
    l: usize,
    b: usize,
    eps: f32,
) -> Vec<f32> {
    let mut y = vec![0.0f32; d * l * b];
    for bi in 0..b {
        for li in 0..l {
            let offset = d * (li + l * bi);
            let mut mean = 0.0f32;
            for i in 0..d {
                mean += x[offset + i];
            }
            mean /= d as f32;
            let mut var = 0.0f32;
            for i in 0..d {
                let centered = x[offset + i] - mean;
                var += centered * centered;
            }
            var /= d as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for i in 0..d {
                let normalized = (x[offset + i] - mean) * inv;
                let mut value = normalized * ln.weight[i];
                if let Some(bias) = &ln.bias {
                    value += bias[i];
                }
                y[offset + i] = value;
            }
        }
    }
    y
}

fn linear_host(
    linear: &LinearWeights,
    in_dim: usize,
    out_dim: usize,
    x: &[f32],
    l: usize,
    b: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim * l * b];
    for bi in 0..b {
        for li in 0..l {
            let x_offset = in_dim * (li + l * bi);
            let y_offset = out_dim * (li + l * bi);
            for o in 0..out_dim {
                let mut acc = match &linear.bias {
                    Some(bias) => bias[o],
                    None => 0.0,
                };
                for i in 0..in_dim {
                    acc += linear.weight[i + in_dim * o] * x[x_offset + i];
                }
                y[y_offset + o] = acc;
            }
        }
    }
    y
}

fn add_assign(lhs: &mut [f32], rhs: &[f32]) {
    for (a, b) in lhs.iter_mut().zip(rhs) {
        *a += b;
    }
}

/// Build an attention key mask `(L_q, L_k, B)` (true = keep) as the tenferro
/// `attention` primitive consumes it, matching `attention_masks` in `model.jl`.
fn build_mask(
    cfg: &EncoderConfig,
    kind: LayerKind,
    mask: &[bool],
    length: usize,
    batch: usize,
) -> Vec<bool> {
    let window = cfg.local_attention / 2;
    let mut keep = vec![false; length * length * batch];
    for b in 0..batch {
        for q in 0..length {
            for k in 0..length {
                let valid_k = mask[k + length * b];
                let valid_q = mask[q + length * b];
                let inside = match kind {
                    LayerKind::FullAttention => true,
                    LayerKind::SlidingAttention => q.abs_diff(k) <= window || !valid_q,
                };
                keep[q + length * (k + length * b)] = valid_k && inside;
            }
        }
    }
    keep
}

fn rope_host(
    x: &[f32],
    hd: usize,
    heads: usize,
    length: usize,
    batch: usize,
    base: f64,
) -> Vec<f32> {
    let half = hd / 2;
    let log_base = (base as f32).log2();
    let mut out = x.to_vec();
    for b in 0..batch {
        for pos in 0..length {
            for head in 0..heads {
                let offset = hd * head + hd * heads * (pos + length * b);
                for i in 0..half {
                    let theta = pos as f32 * (-(i as f32) / (half as f32) * log_base).exp2();
                    let (sin, cos) = theta.sin_cos();
                    let a = x[offset + i];
                    let second = x[offset + half + i];
                    out[offset + i] = a * cos - second * sin;
                    out[offset + half + i] = a * sin + second * cos;
                }
            }
        }
    }
    out
}

/// Multi-head self-attention on `(hd, heads, L, B)` inputs with a
/// `(L_q, L_k, B)` keep mask. Returns `(hd, heads, L, B)`.
#[allow(clippy::too_many_arguments)]
fn attention_host(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    keep: &[bool],
    hd: usize,
    heads: usize,
    length: usize,
    batch: usize,
) -> Vec<f32> {
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0f32; hd * heads * length * batch];
    let mut probs = vec![0.0f32; length];
    for b in 0..batch {
        for head in 0..heads {
            for query in 0..length {
                let q_offset = hd * head + hd * heads * (query + length * b);
                let mut max = f32::NEG_INFINITY;
                for key in 0..length {
                    if !keep[query + length * (key + length * b)] {
                        probs[key] = f32::NEG_INFINITY;
                        continue;
                    }
                    let k_offset = hd * head + hd * heads * (key + length * b);
                    let mut acc = 0.0f32;
                    for i in 0..hd {
                        acc += q[q_offset + i] * k[k_offset + i];
                    }
                    let score = acc * scale;
                    probs[key] = score;
                    if score > max {
                        max = score;
                    }
                }
                let mut sum = 0.0f32;
                for value in probs.iter_mut() {
                    *value = if value.is_finite() {
                        (*value - max).exp()
                    } else {
                        0.0
                    };
                    sum += *value;
                }
                for (key, &prob) in probs.iter().enumerate() {
                    let p = prob / sum;
                    if p == 0.0 {
                        continue;
                    }
                    let v_offset = hd * head + hd * heads * (key + length * b);
                    for i in 0..hd {
                        out[q_offset + i] += v[v_offset + i] * p;
                    }
                }
            }
        }
    }
    out
}

/// Split a fused `(3d, L, B)` projection into `q`, `k`, `v` of `(hd, heads, L, B)`.
fn split_qkv(
    qkv: &[f32],
    hidden: usize,
    heads: usize,
    length: usize,
    batch: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let hd = hidden / heads;
    let total = hidden * length * batch;
    let mut q = vec![0.0f32; total];
    let mut k = vec![0.0f32; total];
    let mut v = vec![0.0f32; total];
    for local in 0..length * batch {
        let base = 3 * hidden * local;
        let head_offset = hd * heads * local;
        for head in 0..heads {
            let src = base + hd * head;
            let dst = head_offset + hd * head;
            q[dst..dst + hd].copy_from_slice(&qkv[src..src + hd]);
            k[dst..dst + hd].copy_from_slice(&qkv[src + hidden..src + hidden + hd]);
            v[dst..dst + hd].copy_from_slice(&qkv[src + 2 * hidden..src + 2 * hidden + hd]);
        }
    }
    (q, k, v)
}

/// Fused qkv projection, RoPE, attention and output projection.
#[allow(clippy::too_many_arguments)]
fn attention_block_host(
    in_proj: &LinearWeights,
    out_proj: &LinearWeights,
    num_heads: usize,
    rope_base: Option<f64>,
    x: &[f32],
    keep: &[bool],
    hidden: usize,
    length: usize,
    batch: usize,
) -> Vec<f32> {
    let hd = hidden / num_heads;
    let qkv = linear_host(in_proj, hidden, 3 * hidden, x, length, batch);
    let (mut q, mut k, v) = split_qkv(&qkv, hidden, num_heads, length, batch);
    if let Some(base) = rope_base {
        q = rope_host(&q, hd, num_heads, length, batch, base);
        k = rope_host(&k, hd, num_heads, length, batch, base);
    }
    let attended = attention_host(&q, &k, &v, keep, hd, num_heads, length, batch);
    // (hd, heads, L, B) -> (hidden, L, B)
    linear_host(out_proj, hidden, hidden, &attended, length, batch)
}

fn gelu_gate_host(u: &[f32], intermediate: usize, columns: usize) -> Vec<f32> {
    let mut g = vec![0.0f32; intermediate * columns];
    for c in 0..columns {
        let base = 2 * intermediate * c;
        for i in 0..intermediate {
            g[intermediate * c + i] = gelu_tanh(u[base + i]) * u[base + intermediate + i];
        }
    }
    g
}

fn encoder_host(
    cfg: &EncoderConfig,
    weights: &ModernBertWeights,
    ids: &[i64],
    mask: &[bool],
    batch: usize,
) -> Result<Vec<f32>> {
    let d = cfg.hidden_size;
    let intermediate = cfg.intermediate_size;
    let length = ids.len() / batch;
    let vocab = weights.tok_embeddings.len() / d;
    let eps = cfg.norm_eps as f32;

    let mut x = vec![0.0f32; d * length * batch];
    for b in 0..batch {
        for l in 0..length {
            let id = ids[l + length * b];
            if id < 0 || id as usize >= vocab {
                return Err(DecisionError::invalid_field(
                    "laya.ids",
                    "token id is outside the vocabulary",
                ));
            }
            let id = id as usize;
            let dst = d * (l + length * b);
            let src = d * id;
            x[dst..dst + d].copy_from_slice(&weights.tok_embeddings[src..src + d]);
        }
    }
    x = layer_norm_host(&weights.embed_norm, &x, d, length, batch, eps);

    for layer in &weights.layers {
        let attn_input = match &layer.attn_norm {
            Some(ln) => layer_norm_host(ln, &x, d, length, batch, eps),
            None => x.clone(),
        };
        let keep = build_mask(cfg, layer.kind, mask, length, batch);
        let attended = attention_block_host(
            &layer.wqkv,
            &layer.wo,
            layer.num_heads,
            Some(layer.rope_base),
            &attn_input,
            &keep,
            d,
            length,
            batch,
        );
        let mut z = x.clone();
        add_assign(&mut z, &attended);

        let hn = layer_norm_host(&layer.mlp_norm, &z, d, length, batch, eps);
        let u = linear_host(&layer.wi, d, 2 * intermediate, &hn, length, batch);
        let g = gelu_gate_host(&u, intermediate, length * batch);
        let down = linear_host(&layer.wo_mlp, intermediate, d, &g, length, batch);
        add_assign(&mut z, &down);
        x = z;
    }

    Ok(layer_norm_host(
        &weights.final_norm,
        &x,
        d,
        length,
        batch,
        eps,
    ))
}

fn head_layer_host(
    layer: &HeadLayerWeights,
    x: &[f32],
    keep: &[bool],
    d: usize,
    length: usize,
    batch: usize,
    eps: f32,
) -> Vec<f32> {
    let ff = layer.linear1.weight.len() / d;
    let n = layer_norm_host(&layer.norm1, x, d, length, batch, eps);
    let attended = attention_block_host(
        &layer.in_proj,
        &layer.out_proj,
        layer.num_heads,
        None,
        &n,
        keep,
        d,
        length,
        batch,
    );
    let mut z = x.to_vec();
    add_assign(&mut z, &attended);
    let hn = layer_norm_host(&layer.norm2, &z, d, length, batch, eps);
    let u = linear_host(&layer.linear1, d, ff, &hn, length, batch);
    let r: Vec<f32> = u.iter().map(|value| relu(*value)).collect();
    let y = linear_host(&layer.linear2, ff, d, &r, length, batch);
    add_assign(&mut z, &y);
    z
}

/// Gather `h[:, marker_pos, :]` from `h` `(d, L, B)` into `(d, K, B)`, clamping
/// negative positions to 0 (they are always masked out).
fn gather_markers_host(
    h: &[f32],
    marker_pos: &[i64],
    k_count: usize,
    length: usize,
    batch: usize,
    d: usize,
) -> Vec<f32> {
    let mut markers = vec![0.0f32; d * k_count * batch];
    for b in 0..batch {
        for j in 0..k_count {
            let pos = marker_pos[j + k_count * b].max(0) as usize;
            let src = d * (pos + length * b);
            let dst = d * (j + k_count * b);
            markers[dst..dst + d].copy_from_slice(&h[src..src + d]);
        }
    }
    markers
}

/// Mask marker logits and build the four pooled scalars per batch.
///
/// Returns the masked `(K, B)` logits and the `(4, B)` features
/// `[top1; top1 - top2; entropy; k/255]`.
fn pool_host(
    raw_logits: &[f32],
    marker_mask: &[bool],
    k_count: usize,
    batch: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut logits = raw_logits.to_vec();
    for b in 0..batch {
        for j in 0..k_count {
            if !marker_mask[j + k_count * b] {
                logits[j + k_count * b] = -1.0e4;
            }
        }
    }
    let mut features = vec![0.0f32; 4 * batch];
    for b in 0..batch {
        let mut max = f32::NEG_INFINITY;
        for j in 0..k_count {
            max = max.max(logits[j + k_count * b]);
        }
        let mut probs = vec![0.0f32; k_count];
        let mut sum = 0.0f32;
        for j in 0..k_count {
            let value = (logits[j + k_count * b] - max).exp();
            probs[j] = value;
            sum += value;
        }
        for value in probs.iter_mut() {
            *value /= sum;
        }
        let mut top1 = f32::NEG_INFINITY;
        let mut top2 = f32::NEG_INFINITY;
        for &value in &probs {
            if value > top1 {
                top2 = top1;
                top1 = value;
            } else if value > top2 {
                top2 = value;
            }
        }
        let mut entropy = 0.0f32;
        for &value in &probs {
            entropy -= value * value.max(1.0e-9).ln();
        }
        let mut count = 0usize;
        for j in 0..k_count {
            if marker_mask[j + k_count * b] {
                count += 1;
            }
        }
        let used = (count.max(2)) as f32;
        entropy /= used.ln();
        features[4 * b] = top1;
        features[4 * b + 1] = top1 - top2;
        features[4 * b + 2] = entropy;
        features[4 * b + 3] = used / 255.0;
    }
    (logits, features)
}

// ------------------------------------------------------------------ references

/// Host reference encoder forward. Returns `final_norm(last layer output)` as a
/// column-major `(d, L, B)` vector.
///
/// `ids` and `mask` are column-major `(L, B)` (the sequence axis is fastest).
pub fn forward_encoder_reference(
    cfg: &EncoderConfig,
    weights: &ModernBertWeights,
    ids: &[i64],
    mask: &[bool],
    batch: usize,
) -> Result<Vec<f32>> {
    if batch == 0 || ids.len() % batch != 0 || ids.is_empty() {
        return Err(DecisionError::invalid_field(
            "laya.ids",
            "ids must be a non-empty (L, B) batch",
        ));
    }
    if mask.len() != ids.len() {
        return Err(DecisionError::invalid_field(
            "laya.mask",
            "mask length must match the token batch",
        ));
    }
    weights.validate(cfg)?;
    encoder_host(cfg, weights, ids, mask, batch)
}

/// Host reference decision-model forward.
///
/// Returns the masked marker logits `(K, B)` and the action logits
/// `(n_actions, B)`, both column-major.
#[allow(clippy::too_many_arguments)]
pub fn forward_reference(
    encoder: &EncoderConfig,
    agent: &AgentConfig,
    weights: &LayaWeights,
    ids: &[i64],
    mask: &[bool],
    marker_pos: &[i64],
    marker_mask: &[bool],
    qtype: &[i64],
) -> Result<(Vec<f32>, Vec<f32>)> {
    if qtype.is_empty() {
        return Err(DecisionError::invalid_field(
            "laya.qtype",
            "the batch must be non-empty",
        ));
    }
    let batch = qtype.len();
    if ids.len() % batch != 0 || ids.is_empty() {
        return Err(DecisionError::invalid_field(
            "laya.ids",
            "ids must be a non-empty (L, B) batch",
        ));
    }
    if mask.len() != ids.len() {
        return Err(DecisionError::invalid_field(
            "laya.mask",
            "mask length must match the token batch",
        ));
    }
    if marker_pos.len() % batch != 0 {
        return Err(DecisionError::invalid_field(
            "laya.marker_pos",
            "marker_pos must be a (K, B) batch",
        ));
    }
    if marker_mask.len() != marker_pos.len() {
        return Err(DecisionError::invalid_field(
            "laya.marker_mask",
            "marker_mask must match marker_pos",
        ));
    }
    weights.validate(encoder, agent)?;
    let d = encoder.hidden_size;
    let eps = encoder.norm_eps as f32;
    let length = ids.len() / batch;
    let k_count = marker_pos.len() / batch;
    if k_count < 2 {
        return Err(DecisionError::invalid_field(
            "laya.marker_pos",
            "at least two marker slots are required",
        ));
    }

    let e = encoder_host(encoder, &weights.encoder, ids, mask, batch)?;

    // Typed head: h = encoder_output + type_emb[:, qtype].
    let mut h = vec![0.0f32; d * length * batch];
    for b in 0..batch {
        let q = qtype[b];
        if !(0..3).contains(&q) {
            return Err(DecisionError::invalid_field(
                "laya.qtype",
                "qtype must be 0 (choice), 1 (score) or 2 (noul)",
            ));
        }
        let type_offset = d * q as usize;
        for l in 0..length {
            for i in 0..d {
                h[i + d * (l + length * b)] =
                    e[i + d * (l + length * b)] + weights.type_emb[type_offset + i];
            }
        }
    }
    let head_keep = build_mask(encoder, LayerKind::FullAttention, mask, length, batch);
    for layer in &weights.head {
        h = head_layer_host(layer, &h, &head_keep, d, length, batch, eps);
    }

    let markers = gather_markers_host(&h, marker_pos, k_count, length, batch, d);
    let s0 = layer_norm_host(&weights.scorer_norm, &markers, d, k_count, batch, eps);
    let s1 = linear_host(&weights.scorer1, d, 1, &s0, k_count, batch);
    let g1: Vec<f32> = s1.iter().map(|value| gelu_tanh(*value)).collect();
    let logits = linear_host(&weights.scorer2, 1, 1, &g1, k_count, batch);

    let (masked_logits, features) = pool_host(&logits, marker_mask, k_count, batch);

    let mut pooled = vec![0.0f32; (d + 4) * batch];
    for b in 0..batch {
        for i in 0..d {
            pooled[i + (d + 4) * b] = h[i + d * length * b];
        }
        for f in 0..4 {
            pooled[d + f + (d + 4) * b] = features[f + 4 * b];
        }
    }

    let action_hidden = weights.act1.weight.len() / (d + 4);
    let a1 = linear_host(&weights.act1, d + 4, action_hidden, &pooled, 1, batch);
    let g2: Vec<f32> = a1.iter().map(|value| gelu_tanh(*value)).collect();
    let action = linear_host(
        &weights.act2,
        action_hidden,
        agent.action_count(),
        &g2,
        1,
        batch,
    );
    Ok((masked_logits, action))
}

// ----------------------------------------------------------- tenferro helpers

fn to_ad_error(error: DecisionError) -> tenferro_ad::Error {
    tenferro_ad::Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
        "laya-infer",
        "config",
        error.to_string(),
    ))
}

fn tensor_col(
    session: &mut EagerSession<'_>,
    shape: Vec<usize>,
    data: &[f32],
) -> tenferro_ad::Result<EagerTensor> {
    session.constant_from(Tensor::from_vec_col_major(shape, data.to_vec())?)
}

fn tensor_col_i64(
    session: &mut EagerSession<'_>,
    shape: Vec<usize>,
    data: &[i64],
) -> tenferro_ad::Result<EagerTensor> {
    session.constant_from(Tensor::from_vec_col_major(shape, data.to_vec())?)
}

fn tensor_bool(
    session: &mut EagerSession<'_>,
    shape: Vec<usize>,
    data: &[bool],
) -> tenferro_ad::Result<EagerTensor> {
    session.constant_from(Tensor::from_vec_col_major(shape, data.to_vec())?)
}

fn extract_col(
    session: &mut EagerSession<'_>,
    tensor: &EagerTensor,
) -> tenferro_ad::Result<Vec<f32>> {
    Ok(session.duplicate_value(tensor)?.as_slice::<f32>()?.to_vec())
}

fn slice_axis(
    session: &mut EagerSession<'_>,
    x: &EagerTensor,
    axis: usize,
    start: usize,
    len: usize,
) -> tenferro_ad::Result<EagerTensor> {
    let rank = x.shape().len();
    let mut starts = vec![0usize; rank];
    let mut limits = x.shape().to_vec();
    let strides = vec![1usize; rank];
    starts[axis] = start;
    limits[axis] = start + len;
    session.slice(
        x,
        SliceConfig {
            starts,
            limits,
            strides,
        },
    )
}

/// LayerNorm over axis 0 of an activation `(d, ...)`, returning the same shape.
fn layer_norm_feature_first(
    session: &mut EagerSession<'_>,
    x: &EagerTensor,
    ln: &LayerNormWeights,
    width: usize,
    eps: f64,
) -> tenferro_ad::Result<EagerTensor> {
    let rank = x.shape().len();
    let mut perm: Vec<usize> = (1..rank).collect();
    perm.push(0);
    let transposed = session.transpose(x, &perm)?;
    let weight = tensor_col(session, vec![width], &ln.weight)?;
    let bias = match &ln.bias {
        Some(bias) => Some(tensor_col(session, vec![width], bias)?),
        None => None,
    };
    let normalized = norm::layer_norm(session, &transposed, &weight, bias.as_ref(), eps)?;
    let mut back: Vec<usize> = vec![rank - 1];
    back.extend(0..rank - 1);
    session.transpose(&normalized, &back)
}

/// Linear over axis 0 of an activation `(in, ...)`, returning `(out, ...)`.
fn linear_feature_first(
    session: &mut EagerSession<'_>,
    x: &EagerTensor,
    linear: &LinearWeights,
    out_dim: usize,
) -> tenferro_ad::Result<EagerTensor> {
    let in_dim = x.shape()[0];
    let weight = tensor_col(session, vec![in_dim, out_dim], &linear.weight)?;
    let contracted = session.dot_general(
        x,
        &weight,
        DotGeneralConfig {
            lhs_contracting_dims: [0].as_slice().into(),
            rhs_contracting_dims: [0].as_slice().into(),
            lhs_batch_dims: [].as_slice().into(),
            rhs_batch_dims: [].as_slice().into(),
        },
    )?;
    let rank = contracted.shape().len();
    let mut perm: Vec<usize> = vec![rank - 1];
    perm.extend(0..rank - 1);
    let y = session.transpose(&contracted, &perm)?;
    match &linear.bias {
        Some(bias) => {
            let bias = tensor_col(session, vec![out_dim], bias)?;
            let mut shape = vec![1usize; rank];
            shape[0] = out_dim;
            let bias = session.reshape(&bias, shape)?;
            let dims: Vec<usize> = (0..rank).collect();
            let bias = session.broadcast_in_dim(&bias, y.shape(), &dims)?;
            session.add(&y, &bias)
        }
        None => Ok(y),
    }
}

fn relu_tensor(
    session: &mut EagerSession<'_>,
    x: &EagerTensor,
) -> tenferro_ad::Result<EagerTensor> {
    let zero = session.constant_from(Tensor::from_vec_col_major(vec![], vec![0.0f32])?)?;
    session.maximum(x, &zero)
}

#[allow(clippy::too_many_arguments)]
fn attention_block_tenferro(
    session: &mut EagerSession<'_>,
    hidden: usize,
    num_heads: usize,
    rope_base: Option<f64>,
    in_proj: &LinearWeights,
    out_proj: &LinearWeights,
    x: &EagerTensor,
    keep: &[bool],
    length: usize,
    batch: usize,
) -> tenferro_ad::Result<EagerTensor> {
    let hd = hidden / num_heads;
    let qkv = linear_feature_first(session, x, in_proj, 3 * hidden)?; // (3d, L, B)
    let qkv = session.reshape(&qkv, vec![hidden, 3, length, batch])?;
    let q = slice_axis(session, &qkv, 1, 0, 1)?;
    let k = slice_axis(session, &qkv, 1, 1, 1)?;
    let v = slice_axis(session, &qkv, 1, 2, 1)?;
    let to_heads = |session: &mut EagerSession<'_>, t: &EagerTensor| {
        let shaped = session.reshape(t, vec![hd, num_heads, length, batch])?;
        session.transpose(&shaped, &[3, 1, 2, 0])
    };
    let q = to_heads(session, &q)?; // (B, H, L, hd)
    let k = to_heads(session, &k)?;
    let v = to_heads(session, &v)?;
    let (q, k) = match rope_base {
        Some(base) => (
            rope::rope_modernbert(session, &q, base)?,
            rope::rope_modernbert(session, &k, base)?,
        ),
        None => (q, k),
    };
    let mask = tensor_bool(session, vec![length, length, batch], keep)?;
    let attended = attention::attention(session, &q, &k, &v, Some(&mask), None)?; // (B,H,L,hd)
    let attended = session.transpose(&attended, &[3, 1, 2, 0])?; // (hd,H,L,B)
    let attended = session.reshape(&attended, vec![hidden, length, batch])?;
    linear_feature_first(session, &attended, out_proj, hidden)
}

fn encoder_tensor(
    session: &mut EagerSession<'_>,
    cfg: &EncoderConfig,
    weights: &ModernBertWeights,
    ids: &[i64],
    mask: &[bool],
    batch: usize,
) -> tenferro_ad::Result<EagerTensor> {
    let d = cfg.hidden_size;
    let intermediate = cfg.intermediate_size;
    let length = ids.len() / batch;
    let vocab = weights.tok_embeddings.len() / d;
    let eps = cfg.norm_eps;

    let table = tensor_col(session, vec![d, vocab], &weights.tok_embeddings)?;
    let ids_t = tensor_col_i64(session, vec![length, batch], ids)?;
    let mut x = session.gather(
        &table,
        &ids_t,
        GatherConfig {
            offset_dims: vec![0],
            collapsed_slice_dims: vec![1],
            start_index_map: vec![1],
            index_vector_dim: 2,
            slice_sizes: vec![d, 1],
        },
    )?; // (d, L, B)
    x = layer_norm_feature_first(session, &x, &weights.embed_norm, d, eps)?;

    for layer in &weights.layers {
        let attn_input = match &layer.attn_norm {
            Some(ln) => layer_norm_feature_first(session, &x, ln, d, eps)?,
            None => x.clone(),
        };
        let keep = build_mask(cfg, layer.kind, mask, length, batch);
        let attended = attention_block_tenferro(
            session,
            d,
            layer.num_heads,
            Some(layer.rope_base),
            &layer.wqkv,
            &layer.wo,
            &attn_input,
            &keep,
            length,
            batch,
        )?;
        let z = session.add(&x, &attended)?;

        let hn = layer_norm_feature_first(session, &z, &layer.mlp_norm, d, eps)?;
        let u = linear_feature_first(session, &hn, &layer.wi, 2 * intermediate)?;
        let value = slice_axis(session, &u, 0, 0, intermediate)?;
        let gate = slice_axis(session, &u, 0, intermediate, intermediate)?;
        let g = activation::geglu(session, &value, &gate)?;
        let down = linear_feature_first(session, &g, &layer.wo_mlp, d)?;
        x = session.add(&z, &down)?;
    }

    layer_norm_feature_first(session, &x, &weights.final_norm, d, eps)
}

#[allow(clippy::too_many_arguments)]
fn head_layer_tenferro(
    session: &mut EagerSession<'_>,
    cfg: &EncoderConfig,
    layer: &HeadLayerWeights,
    x: &EagerTensor,
    keep: &[bool],
    length: usize,
    batch: usize,
) -> tenferro_ad::Result<EagerTensor> {
    let d = cfg.hidden_size;
    let eps = cfg.norm_eps;
    let n = layer_norm_feature_first(session, x, &layer.norm1, d, eps)?;
    let attended = attention_block_tenferro(
        session,
        d,
        layer.num_heads,
        None,
        &layer.in_proj,
        &layer.out_proj,
        &n,
        keep,
        length,
        batch,
    )?;
    let z = session.add(x, &attended)?;
    let hn = layer_norm_feature_first(session, &z, &layer.norm2, d, eps)?;
    let ff = layer.linear1.weight.len() / d;
    let u = linear_feature_first(session, &hn, &layer.linear1, ff)?;
    let r = relu_tensor(session, &u)?;
    let y = linear_feature_first(session, &r, &layer.linear2, d)?;
    session.add(&z, &y)
}

fn gather_type_emb(
    session: &mut EagerSession<'_>,
    type_emb: &[f32],
    qtype: &[i64],
    d: usize,
    batch: usize,
) -> tenferro_ad::Result<EagerTensor> {
    let table = tensor_col(session, vec![d, 3], type_emb)?;
    let indices = tensor_col_i64(session, vec![batch], qtype)?;
    session.gather(
        &table,
        &indices,
        GatherConfig {
            offset_dims: vec![0],
            collapsed_slice_dims: vec![1],
            start_index_map: vec![1],
            index_vector_dim: 1,
            slice_sizes: vec![d, 1],
        },
    )
}

fn gather_markers(
    session: &mut EagerSession<'_>,
    h: &EagerTensor,
    marker_pos: &[i64],
    k_count: usize,
    length: usize,
    batch: usize,
    d: usize,
) -> tenferro_ad::Result<EagerTensor> {
    let flat = session.reshape(h, vec![d, length * batch])?;
    let mut indices = vec![0i64; k_count * batch];
    for b in 0..batch {
        for j in 0..k_count {
            let pos = marker_pos[j + k_count * b].max(0) as usize;
            indices[j + k_count * b] = (pos + length * b) as i64;
        }
    }
    let indices = tensor_col_i64(session, vec![k_count, batch], &indices)?;
    session.gather(
        &flat,
        &indices,
        GatherConfig {
            offset_dims: vec![0],
            collapsed_slice_dims: vec![1],
            start_index_map: vec![1],
            index_vector_dim: 2,
            slice_sizes: vec![d, 1],
        },
    )
}

// ------------------------------------------------------------------- tenferro

/// Tenferro-backed encoder forward. Returns a column-major `(d, L, B)` vector.
pub fn forward_encoder_tenferro(
    session: &mut EagerSession<'_>,
    cfg: &EncoderConfig,
    weights: &ModernBertWeights,
    ids: &[i64],
    mask: &[bool],
    batch: usize,
) -> tenferro_ad::Result<Vec<f32>> {
    if batch == 0 || ids.len() % batch != 0 || ids.is_empty() {
        return Err(to_ad_error(DecisionError::invalid_field(
            "laya.ids",
            "ids must be a non-empty (L, B) batch",
        )));
    }
    if mask.len() != ids.len() {
        return Err(to_ad_error(DecisionError::invalid_field(
            "laya.mask",
            "mask length must match the token batch",
        )));
    }
    weights.validate(cfg).map_err(to_ad_error)?;
    let h = encoder_tensor(session, cfg, weights, ids, mask, batch)?;
    extract_col(session, &h)
}

/// Tenferro-backed decision-model forward.
///
/// Returns the masked marker logits `(K, B)` and the action logits
/// `(n_actions, B)`, both column-major. The encoder, typed head and marker
/// scorer run through the session; the marker softmax/top-k pooling follows the
/// Julia reference and is computed on the host, then the action head runs
/// through the session again.
#[allow(clippy::too_many_arguments)]
pub fn forward_tenferro(
    session: &mut EagerSession<'_>,
    encoder: &EncoderConfig,
    agent: &AgentConfig,
    weights: &LayaWeights,
    ids: &[i64],
    mask: &[bool],
    marker_pos: &[i64],
    marker_mask: &[bool],
    qtype: &[i64],
) -> tenferro_ad::Result<(Vec<f32>, Vec<f32>)> {
    if qtype.is_empty() {
        return Err(to_ad_error(DecisionError::invalid_field(
            "laya.qtype",
            "the batch must be non-empty",
        )));
    }
    let batch = qtype.len();
    if ids.len() % batch != 0 || ids.is_empty() {
        return Err(to_ad_error(DecisionError::invalid_field(
            "laya.ids",
            "ids must be a non-empty (L, B) batch",
        )));
    }
    if mask.len() != ids.len() {
        return Err(to_ad_error(DecisionError::invalid_field(
            "laya.mask",
            "mask length must match the token batch",
        )));
    }
    if marker_pos.len() % batch != 0 || marker_mask.len() != marker_pos.len() {
        return Err(to_ad_error(DecisionError::invalid_field(
            "laya.marker_pos",
            "marker_pos and marker_mask must be (K, B) batches",
        )));
    }
    weights.validate(encoder, agent).map_err(to_ad_error)?;
    let d = encoder.hidden_size;
    let length = ids.len() / batch;
    let k_count = marker_pos.len() / batch;
    if k_count < 2 {
        return Err(to_ad_error(DecisionError::invalid_field(
            "laya.marker_pos",
            "at least two marker slots are required",
        )));
    }

    let encoder_output = encoder_tensor(session, encoder, &weights.encoder, ids, mask, batch)?;
    let te = gather_type_emb(session, &weights.type_emb, qtype, d, batch)?;
    let te = session.broadcast_in_dim(&te, &[d, length, batch], &[0, 2])?;
    let mut h = session.add(&encoder_output, &te)?;

    let head_keep = build_mask(encoder, LayerKind::FullAttention, mask, length, batch);
    for layer in &weights.head {
        h = head_layer_tenferro(session, encoder, layer, &h, &head_keep, length, batch)?;
    }

    let markers = gather_markers(session, &h, marker_pos, k_count, length, batch, d)?;
    let s0 =
        layer_norm_feature_first(session, &markers, &weights.scorer_norm, d, encoder.norm_eps)?;
    let s1 = linear_feature_first(session, &s0, &weights.scorer1, 1)?;
    let g1 = activation::gelu(session, &s1)?;
    let s2 = linear_feature_first(session, &g1, &weights.scorer2, 1)?;
    let s2 = session.reshape(&s2, vec![k_count, batch])?;
    let raw_logits = extract_col(session, &s2)?;
    let (logits, features) = pool_host(&raw_logits, marker_mask, k_count, batch);

    let first = slice_axis(session, &h, 1, 0, 1)?;
    let first = session.reshape(&first, vec![d, batch])?;
    let first = extract_col(session, &first)?;
    let mut pooled = vec![0.0f32; (d + 4) * batch];
    for b in 0..batch {
        for i in 0..d {
            pooled[i + (d + 4) * b] = first[i + d * b];
        }
        for f in 0..4 {
            pooled[d + f + (d + 4) * b] = features[f + 4 * b];
        }
    }
    let pooled = tensor_col(session, vec![d + 4, batch], &pooled)?;
    let action_hidden = weights.act1.weight.len() / (d + 4);
    let a1 = linear_feature_first(session, &pooled, &weights.act1, action_hidden)?;
    let g2 = activation::gelu(session, &a1)?;
    let action = linear_feature_first(session, &g2, &weights.act2, agent.action_count())?;
    let action = extract_col(session, &action)?;
    Ok((logits, action))
}
