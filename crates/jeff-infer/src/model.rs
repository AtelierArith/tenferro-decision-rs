//! Jeff Qwen3.5 text-model layer stack.
//!
//! Composes embeddings, per-layer input RMSNorm, full attention or Gated
//! DeltaNet, the residual + post RMSNorm, the SiLU-gated MLP, the final
//! RMSNorm (last position only), and the trained readout.
//!
//! Two forward implementations are provided:
//!
//! - [`forward_reference`]: all host, running the Gated DeltaNet layers through
//!   the fused recurrent kernel ([`delta_layer_recurrent`]).
//! - [`forward_tenferro`]: embedding, norms, full attention, the DeltaNet
//!   layer, the MLP, and the readout run through the eager session. The
//!   DeltaNet layer goes through the [`gated_delta`] plan-dispatched entry.
//!
//! Both have `_with` variants that take a reusable [`GatedDeltaWorkspace`] so a
//! caller (e.g. an engine answering many rows) can avoid reallocating the
//! scratch buffers.
//!
//! Full-attention weights are assumed **GQA-expanded** (one k/v head per query
//! head) at preparation time, and the fused `q`/gate projection is split into
//! separate `q` and `gate` weights. This keeps the runtime layout uniform.

use decision_core::{DecisionError, Result};
use tenferro_ad::{EagerSession, EagerTensor};
use tenferro_ext::{
    EagerSessionGatedSiluExt, EagerSessionJeffAttentionExt, EagerSessionLinearExt,
    EagerSessionRmsNormExt,
};
use tenferro_gated_delta::{
    EagerSessionGatedDeltaExt, GatedDeltaConfig, GatedDeltaOp, GatedDeltaWeights,
    GatedDeltaWorkspace, delta_layer_recurrent, delta_layer_tenferro_cached,
    prepare_kernel_weights, prepare_tensor_weights,
};
use tenferro_infer::{TensorCache, activation, embedding, norm, rope};

/// How the tenferro forward runs a Gated DeltaNet layer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeltaKernel {
    /// The fused host recurrent kernel, invoked through the `GatedDelta`
    /// extension op (CPU only). The session stays tenferro-first; only the
    /// layer's kernel is the host fast path.
    #[default]
    HostRecurrent,
    /// The fully tensor-native chunked formulation. Backend-portable, but ~1.8x
    /// slower on CPU.
    TensorNative,
}

/// Shared model dimensions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JeffConfig {
    /// Hidden width.
    pub hidden: usize,
    /// Number of attention heads.
    pub heads: usize,
    /// Attention head width.
    pub head_dim: usize,
    /// MLP intermediate width.
    pub intermediate: usize,
    /// RMSNorm epsilon.
    pub eps: f32,
}

/// Full-attention weights (GQA already expanded to `heads`; `q` and `gate`
/// already split).
#[derive(Clone, Debug)]
pub struct FullAttentionWeights {
    /// Query projection `(hidden, heads*head_dim)`.
    pub q: Vec<f32>,
    /// Query gate projection `(hidden, heads*head_dim)`.
    pub gate: Vec<f32>,
    /// Key projection `(hidden, heads*head_dim)`.
    pub k: Vec<f32>,
    /// Value projection `(hidden, heads*head_dim)`.
    pub v: Vec<f32>,
    /// Output projection `(heads*head_dim, hidden)`.
    pub o: Vec<f32>,
    /// Query RMSNorm scale `(head_dim,)`.
    pub q_norm: Vec<f32>,
    /// Key RMSNorm scale `(head_dim,)`.
    pub k_norm: Vec<f32>,
    /// RoPE base.
    pub rope_theta: f32,
    /// Number of rotated channels per head.
    pub rotary_dim: usize,
}

/// SiLU-gated MLP weights.
#[derive(Clone, Debug)]
pub struct MlpWeights {
    /// Gate projection `(hidden, intermediate)`.
    pub gate: Vec<f32>,
    /// Up projection `(hidden, intermediate)`.
    pub up: Vec<f32>,
    /// Down projection `(intermediate, hidden)`.
    pub down: Vec<f32>,
}

/// One layer's attention block.
#[derive(Clone, Debug)]
pub enum AttentionWeights {
    /// Full (softmax) attention.
    Full(FullAttentionWeights),
    /// Gated DeltaNet linear attention.
    Delta {
        /// Prepared DeltaNet weights.
        weights: GatedDeltaWeights,
        /// DeltaNet shape/execution settings.
        config: GatedDeltaConfig,
    },
}

/// One decoder layer.
#[derive(Clone, Debug)]
pub struct LayerWeights {
    /// Input RMSNorm scale `(hidden,)`.
    pub input_norm: Vec<f32>,
    /// Post-attention RMSNorm scale `(hidden,)`.
    pub post_norm: Vec<f32>,
    /// Attention block.
    pub attention: AttentionWeights,
    /// MLP block.
    pub mlp: MlpWeights,
}

/// The full model weights (row-major `(in, out)` projections).
#[derive(Clone, Debug)]
pub struct JeffWeights {
    /// Embedding table `(hidden, vocab)`.
    pub embedding: Vec<f32>,
    /// Vocabulary size.
    pub vocab: usize,
    /// Decoder layers.
    pub layers: Vec<LayerWeights>,
    /// Final RMSNorm scale `(hidden,)`.
    pub final_norm: Vec<f32>,
    /// Readout `(hidden, options)`.
    pub readout: Vec<f32>,
    /// Number of readout columns.
    pub options: usize,
}

impl JeffWeights {
    /// Validate the top-level shapes.
    pub fn validate(&self, cfg: &JeffConfig) -> Result<()> {
        if self.embedding.len() != cfg.hidden * self.vocab {
            return Err(DecisionError::invalid_field(
                "jeff.embedding",
                "embedding shape does not match (hidden, vocab)",
            ));
        }
        if self.final_norm.len() != cfg.hidden {
            return Err(DecisionError::invalid_field(
                "jeff.final_norm",
                "final_norm must have hidden values",
            ));
        }
        if self.readout.len() != cfg.hidden * self.options {
            return Err(DecisionError::invalid_field(
                "jeff.readout",
                "readout shape does not match (hidden, options)",
            ));
        }
        if self.layers.is_empty() {
            return Err(DecisionError::invalid_field(
                "jeff.layers",
                "at least one layer is required",
            ));
        }
        Ok(())
    }
}

// ----------------------------------------------------------------- host helpers

fn linear_host(
    weight: &[f32],
    in_dim: usize,
    out_dim: usize,
    x: &[f32],
    length: usize,
) -> Vec<f32> {
    cpu_kernels::matmul_row_major(weight, in_dim, out_dim, x, length)
}

/// Dense `y = x · W` on the eager session, using the CPU `linear` extension
/// for F32 CPU sessions and native `dot_general` otherwise. Its weight tensors are cached with
/// [`TensorCache::col_major`] over the raw safetensors buffer, so the op reads
/// the natural row-major `(in, out)` weight storage (the fast `matrixmultiply`
/// orientation); see `cpu_kernels::matmul_col_major_into`.
fn linear_tenferro(
    session: &mut EagerSession<'_>,
    x: &EagerTensor,
    weight: &EagerTensor,
) -> tenferro_ad::Result<EagerTensor> {
    if x.dtype() == tenferro_tensor::DType::F32 && tenferro_ext::cpu_extensions_supported(session) {
        session.linear(x, weight)
    } else {
        tenferro_infer::linear::linear(session, x, weight)
    }
}

/// Feature-last RMSNorm on the eager session: the `cpu-kernels` extension op
/// for `f32` rank-2 activations, else the composed `norm::rms_norm`.
fn rms_norm_tenferro(
    session: &mut EagerSession<'_>,
    x: &EagerTensor,
    weight: &EagerTensor,
    centered: bool,
    eps: f64,
) -> tenferro_ad::Result<EagerTensor> {
    if x.shape().len() == 2
        && x.dtype() == tenferro_tensor::DType::F32
        && tenferro_ext::cpu_extensions_supported(session)
    {
        session.rms_norm_last(x, weight, centered, eps)
    } else {
        norm::rms_norm(session, x, weight, centered, eps)
    }
}

/// `silu(gate) * up`: the `cpu-kernels` extension op for `f32`, else the
/// composed activation.
fn gated_silu_tenferro(
    session: &mut EagerSession<'_>,
    gate: &EagerTensor,
    up: &EagerTensor,
) -> tenferro_ad::Result<EagerTensor> {
    if gate.dtype() == tenferro_tensor::DType::F32
        && tenferro_ext::cpu_extensions_supported(session)
    {
        session.gated_silu(gate, up)
    } else {
        activation::gated_silu(session, gate, up)
    }
}

fn rms_centered_rows(
    x: &[f32],
    weight: &[f32],
    hidden: usize,
    length: usize,
    eps: f32,
) -> Vec<f32> {
    let inv_hidden = 1.0 / hidden as f32;
    let mut y = vec![0.0f32; hidden * length];
    for t in 0..length {
        let mean_sq: f32 = (0..hidden).map(|d| x[d * length + t].powi(2)).sum::<f32>() * inv_hidden;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        for d in 0..hidden {
            y[d * length + t] = x[d * length + t] * scale * (1.0 + weight[d]);
        }
    }
    y
}

fn rope_partial_host(
    x: &[f32],
    head_dim: usize,
    heads: usize,
    length: usize,
    rotary_dim: usize,
    theta: f32,
) -> Vec<f32> {
    let half = rotary_dim / 2;
    let mut out = x.to_vec();
    for head in 0..heads {
        for t in 0..length {
            for i in 0..half {
                let angle = (t as f32) * theta.powf(-2.0 * i as f32 / rotary_dim as f32);
                let (cos, sin) = (angle.cos(), angle.sin());
                let first = (head * head_dim + i) * length + t;
                let second = (head * head_dim + i + half) * length + t;
                let a = x[first];
                let b = x[second];
                out[first] = a * cos - b * sin;
                out[second] = a * sin + b * cos;
            }
        }
    }
    out
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn full_attention_host(
    w: &FullAttentionWeights,
    cfg: &JeffConfig,
    x: &[f32],
    mask: &[f32],
) -> Vec<f32> {
    let length = mask.len();
    let hd = cfg.head_dim;
    let heads = cfg.heads;
    let width = hd * heads;

    // Projections, each stored (width, length) with row = d + hd*head.
    let q = linear_host(&w.q, cfg.hidden, width, x, length);
    let gate = linear_host(&w.gate, cfg.hidden, width, x, length);
    let k = linear_host(&w.k, cfg.hidden, width, x, length);
    let v = linear_host(&w.v, cfg.hidden, width, x, length);

    // RMSNorm per (head, position) over the head dimension, then RoPE.
    let q_normed = rms_heads(&q, hd, heads, length, &w.q_norm, cfg.eps);
    let k_normed = rms_heads(&k, hd, heads, length, &w.k_norm, cfg.eps);
    let q_rope = rope_partial_host(&q_normed, hd, heads, length, w.rotary_dim, w.rope_theta);
    let k_rope = rope_partial_host(&k_normed, hd, heads, length, w.rotary_dim, w.rope_theta);

    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0f32; width * length];
    let mut scores = vec![0.0f32; length];
    for head in 0..heads {
        for query in 0..length {
            let mut max = f32::NEG_INFINITY;
            for key in 0..length {
                let keep = key <= query && mask[key] != 0.0;
                let value = if keep {
                    let mut acc = 0.0f32;
                    for d in 0..hd {
                        let row = head * hd + d;
                        acc += k_rope[row * length + key] * q_rope[row * length + query];
                    }
                    acc * scale
                } else {
                    f32::NEG_INFINITY
                };
                scores[key] = value;
                max = max.max(value);
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
            for key in 0..length {
                let p = scores[key] / sum;
                for d in 0..hd {
                    let row = head * hd + d;
                    out[row * length + query] += v[row * length + key] * p;
                }
            }
            // Query gate.
            for d in 0..hd {
                let row = head * hd + d;
                out[row * length + query] *= sigmoid(gate[row * length + query]);
            }
        }
    }

    linear_host(&w.o, width, cfg.hidden, &out, length)
}

/// RMSNorm over the head dimension of a `(width, length)` row-major activation
/// where `width = head_dim * heads`.
fn rms_heads(
    x: &[f32],
    head_dim: usize,
    heads: usize,
    length: usize,
    weight: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut out = x.to_vec();
    for head in 0..heads {
        for t in 0..length {
            let mut mean_sq = 0.0f32;
            for d in 0..head_dim {
                let row = head * head_dim + d;
                mean_sq += x[row * length + t].powi(2);
            }
            mean_sq /= head_dim as f32;
            let scale = 1.0 / (mean_sq + eps).sqrt();
            for (d, w) in weight.iter().enumerate() {
                let row = head * head_dim + d;
                out[row * length + t] = x[row * length + t] * scale * (1.0 + w);
            }
        }
    }
    out
}

fn mlp_host(mlp: &MlpWeights, cfg: &JeffConfig, x: &[f32], length: usize) -> Vec<f32> {
    let gate = linear_host(&mlp.gate, cfg.hidden, cfg.intermediate, x, length);
    let up = linear_host(&mlp.up, cfg.hidden, cfg.intermediate, x, length);
    let gated: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
        .collect();
    linear_host(&mlp.down, cfg.intermediate, cfg.hidden, &gated, length)
}

// ------------------------------------------------------------------ reference

/// Full host reference forward. Returns readout scores `(options,)`.
pub fn forward_reference(
    cfg: &JeffConfig,
    weights: &JeffWeights,
    ids: &[i64],
    mask: &[f32],
) -> Result<Vec<f32>> {
    let mut workspace = GatedDeltaWorkspace::new();
    forward_reference_with(&mut workspace, cfg, weights, ids, mask)
}

/// [`forward_reference`] with a reusable DeltaNet workspace.
///
/// A single workspace is shared across every Gated DeltaNet layer in the stack.
pub fn forward_reference_with(
    workspace: &mut GatedDeltaWorkspace,
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

    let mut hidden = vec![0.0f32; cfg.hidden * length];
    for (t, id) in ids.iter().enumerate() {
        let id = *id as usize;
        if id >= weights.vocab {
            return Err(DecisionError::invalid_field(
                "jeff.ids",
                "token id is outside the vocabulary",
            ));
        }
        for d in 0..cfg.hidden {
            hidden[d * length + t] = weights.embedding[d * weights.vocab + id];
        }
    }

    for layer in &weights.layers {
        let normalized = rms_centered_rows(&hidden, &layer.input_norm, cfg.hidden, length, cfg.eps);
        let mixed = match &layer.attention {
            AttentionWeights::Full(w) => full_attention_host(w, cfg, &normalized, mask),
            AttentionWeights::Delta { weights, config } => {
                delta_layer_recurrent(config, weights, &normalized, mask, workspace)?.to_vec()
            }
        };
        let residual: Vec<f32> = hidden.iter().zip(&mixed).map(|(a, b)| a + b).collect();
        let normalized2 =
            rms_centered_rows(&residual, &layer.post_norm, cfg.hidden, length, cfg.eps);
        let mlp = mlp_host(&layer.mlp, cfg, &normalized2, length);
        hidden = residual.iter().zip(&mlp).map(|(a, b)| a + b).collect();
    }

    // Final norm on the last position, then the readout.
    let mut last = vec![0.0f32; cfg.hidden];
    for d in 0..cfg.hidden {
        last[d] = hidden[d * length + (length - 1)];
    }
    let mean_sq: f32 = last.iter().map(|v| v * v).sum::<f32>() / cfg.hidden as f32;
    let scale = 1.0 / (mean_sq + cfg.eps).sqrt();
    let mut logits = vec![0.0f32; weights.options];
    for (o, logit) in logits.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (d, (value, norm)) in last.iter().zip(&weights.final_norm).enumerate() {
            let nf = value * scale * (1.0 + norm);
            acc += weights.readout[d * weights.options + o] * nf;
        }
        *logit = acc;
    }
    Ok(logits)
}

// ------------------------------------------------------------------ tenferro

#[cfg(test)]
fn host_to_tensor(
    session: &mut EagerSession<'_>,
    rows: usize,
    cols: usize,
    row_major: &[f32],
) -> tenferro_ad::Result<EagerTensor> {
    let mut col = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            col[r + c * rows] = row_major[r * cols + c];
        }
    }
    session.constant_from_host(tenferro_ad::Tensor::from_vec_col_major(
        vec![rows, cols],
        col,
    )?)
}

fn extract(
    session: &mut EagerSession<'_>,
    tensor: &EagerTensor,
    rows: usize,
    cols: usize,
) -> tenferro_ad::Result<Vec<f32>> {
    let host = tenferro_infer::output::host_value(session, tensor)?;
    let values = host.as_slice::<f32>()?;
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] = values[r + c * rows];
        }
    }
    Ok(out)
}

#[cfg(test)]
fn linear_col(
    session: &mut EagerSession<'_>,
    x: &EagerTensor,
    weight: &EagerTensor,
) -> tenferro_ad::Result<EagerTensor> {
    let y = session.dot_general(
        x,
        weight,
        tenferro_ad::DotGeneralConfig {
            lhs_contracting_dims: [0].as_slice().into(),
            rhs_contracting_dims: [0].as_slice().into(),
            lhs_batch_dims: [].as_slice().into(),
            rhs_batch_dims: [].as_slice().into(),
        },
    )?;
    session.transpose(&y, &[1, 0])
}

/// Full attention over `x` in `(length, hidden)` orientation, returning
/// `(length, hidden)`.
fn full_attention_tenferro(
    session: &mut EagerSession<'_>,
    cache: &mut TensorCache,
    w: &FullAttentionWeights,
    cfg: &JeffConfig,
    x: &EagerTensor,
    mask: &[f32],
    length: usize,
) -> tenferro_ad::Result<EagerTensor> {
    let hd = cfg.head_dim;
    let heads = cfg.heads;
    let width = hd * heads;

    // Linear weights are cached with `col_major` (raw row-major `(in, out)`
    // storage) because the `linear` extension op reads that storage directly;
    // see `linear_tenferro`.
    let q_w = cache.col_major(session, vec![cfg.hidden, width], &w.q)?;
    let gate_w = cache.col_major(session, vec![cfg.hidden, width], &w.gate)?;
    let k_w = cache.col_major(session, vec![cfg.hidden, width], &w.k)?;
    let v_w = cache.col_major(session, vec![cfg.hidden, width], &w.v)?;
    let o_w = cache.col_major(session, vec![width, cfg.hidden], &w.o)?;
    let q_norm = cache.col(session, vec![hd, 1], &w.q_norm)?;
    let k_norm = cache.col(session, vec![hd, 1], &w.k_norm)?;
    let q_norm = session.reshape(&q_norm, vec![hd])?;
    let k_norm = session.reshape(&k_norm, vec![hd])?;

    // Linear projections via the `linear` extension op.
    let q = linear_tenferro(session, x, &q_w)?; // (length, width)
    let k = linear_tenferro(session, x, &k_w)?;
    let v = linear_tenferro(session, x, &v_w)?;
    let gate = linear_tenferro(session, x, &gate_w)?;

    let merged = if x.dtype() == tenferro_tensor::DType::F32
        && tenferro_ext::cpu_extensions_supported(session)
    {
        // Fused RMSNorm ×2 + partial RoPE ×2 + causal masked attention + gate.
        let active = session.constant_from_host(tenferro_ad::Tensor::from_vec_col_major(
            vec![length],
            mask.to_vec(),
        )?)?;
        session.jeff_full_attention(
            &q,
            &k,
            &v,
            &gate,
            &q_norm,
            &k_norm,
            &active,
            heads,
            hd,
            w.rotary_dim,
            w.rope_theta as f64,
            cfg.eps as f64,
        )?
    } else {
        // Non-f32 fallback: the eager op sequence.
        let to_heads = |session: &mut EagerSession<'_>, t: &EagerTensor| {
            let shaped = session.reshape(t, vec![length, hd, heads])?;
            session.transpose(&shaped, &[2, 0, 1])
        };
        let q = to_heads(session, &q)?;
        let q = rms_norm_tenferro(session, &q, &q_norm, true, cfg.eps as f64)?;
        let k = to_heads(session, &k)?;
        let k = rms_norm_tenferro(session, &k, &k_norm, true, cfg.eps as f64)?;
        let v = to_heads(session, &v)?;
        let q = rope::rope_qwen_partial(session, &q, w.rope_theta as f64, w.rotary_dim)?;
        let k = rope::rope_qwen_partial(session, &k, w.rope_theta as f64, w.rotary_dim)?;
        let q = session.reshape(&q, vec![1, heads, length, hd])?;
        let k = session.reshape(&k, vec![1, heads, length, hd])?;
        let v = session.reshape(&v, vec![1, heads, length, hd])?;
        let mut keep = vec![false; length * length];
        for t in 0..length {
            for s in 0..length {
                keep[t * length + s] = s <= t && mask[s] != 0.0;
            }
        }
        let mask_t = {
            let mut col = vec![false; length * length];
            for r in 0..length {
                for c in 0..length {
                    col[r + c * length] = keep[r * length + c];
                }
            }
            if tenferro_ext::cpu_extensions_supported(session) {
                session.constant_from_host(tenferro_ad::Tensor::from_vec_col_major(
                    vec![length, length],
                    col,
                )?)?
            } else {
                tenferro_infer::input::bool_tensor_native(session, vec![length, length], &col)?
            }
        };
        let attended =
            tenferro_infer::attention::attention(session, &q, &k, &v, Some(&mask_t), None)?;
        let attended = session.reshape(&attended, vec![heads, length, hd])?;
        let gate = to_heads(session, &gate)?;
        let gate = activation::sigmoid(session, &gate)?;
        let gated = session.mul(&attended, &gate)?;
        let merged = session.transpose(&gated, &[1, 2, 0])?;
        session.reshape(&merged, vec![length, width])?
    };

    let out = linear_tenferro(session, &merged, &o_w)?; // (length, hidden)
    Ok(out)
}

/// Tenferro-backed forward. Returns readout scores `(options,)`.
pub fn forward_tenferro(
    session: &mut EagerSession<'_>,
    cfg: &JeffConfig,
    weights: &JeffWeights,
    ids: &[i64],
    mask: &[f32],
) -> tenferro_ad::Result<Vec<f32>> {
    let mut workspace = GatedDeltaWorkspace::new();
    forward_tenferro_with(&mut workspace, session, cfg, weights, ids, mask)
}

/// [`forward_tenferro`] with a reusable DeltaNet workspace.
///
/// The DeltaNet layers dispatch through the [`gated_delta`] plan entry, so the
/// resolved algorithm comes from each layer's [`GatedDeltaConfig`].
pub fn forward_tenferro_with(
    workspace: &mut GatedDeltaWorkspace,
    session: &mut EagerSession<'_>,
    cfg: &JeffConfig,
    weights: &JeffWeights,
    ids: &[i64],
    mask: &[f32],
) -> tenferro_ad::Result<Vec<f32>> {
    let mut cache = TensorCache::new();
    forward_tenferro_cached(workspace, &mut cache, session, cfg, weights, ids, mask)
}

/// [`forward_tenferro_with`] with a reusable DeltaNet workspace **and** weight
/// cache, so the weights are not re-transposed and re-created on every call.
pub fn forward_tenferro_cached(
    workspace: &mut GatedDeltaWorkspace,
    cache: &mut TensorCache,
    session: &mut EagerSession<'_>,
    cfg: &JeffConfig,
    weights: &JeffWeights,
    ids: &[i64],
    mask: &[f32],
) -> tenferro_ad::Result<Vec<f32>> {
    forward_tenferro_cached_kernel(
        workspace,
        cache,
        session,
        cfg,
        weights,
        ids,
        mask,
        DeltaKernel::default(),
    )
}

/// [`forward_tenferro_cached`] with an explicit [`DeltaKernel`].
#[allow(clippy::too_many_arguments)]
pub fn forward_tenferro_cached_kernel(
    workspace: &mut GatedDeltaWorkspace,
    cache: &mut TensorCache,
    session: &mut EagerSession<'_>,
    cfg: &JeffConfig,
    weights: &JeffWeights,
    ids: &[i64],
    mask: &[f32],
    kernel: DeltaKernel,
) -> tenferro_ad::Result<Vec<f32>> {
    weights.validate(cfg).map_err(config_error)?;
    let length = ids.len();
    if mask.len() != length {
        return Err(config_error(DecisionError::invalid_field(
            "jeff.mask",
            "mask length must match the token length",
        )));
    }

    // Embedding: gather rows of (vocab, hidden) by token ids. The row-major
    // `(hidden, vocab)` table is already the column-major `(vocab, hidden)`
    // tensor, so no transpose is needed. `embedding` returns `(length, hidden)`,
    // the orientation the whole forward stays in — rms_norm normalizes the last
    // axis and `linear` contracts the last axis, so no per-layer transposes.
    let table = cache.col_major(session, vec![weights.vocab, cfg.hidden], &weights.embedding)?;
    let ids_t = session.constant_from_host(tenferro_ad::Tensor::from_vec_col_major(
        vec![length],
        ids.to_vec(),
    )?)?;
    let mut hidden = embedding::embedding(session, &table, &ids_t)?; // (length, hidden)
    // Share the request mask across all Delta layers and both formulations.
    let mask_t = session.constant_from_host(tenferro_ad::Tensor::from_vec_col_major(
        vec![length],
        mask.to_vec(),
    )?)?;

    for layer in &weights.layers {
        let input_norm = cache.col(session, vec![cfg.hidden, 1], &layer.input_norm)?;
        let input_norm = session.reshape(&input_norm, vec![cfg.hidden])?;
        let normalized = rms_norm_tenferro(session, &hidden, &input_norm, true, cfg.eps as f64)?;

        let mixed = match &layer.attention {
            AttentionWeights::Full(w) => {
                full_attention_tenferro(session, cache, w, cfg, &normalized, mask, length)?
            }
            AttentionWeights::Delta { weights, config } => match kernel {
                DeltaKernel::TensorNative => {
                    // The chunked kernel is written for `(hidden, length)`.
                    let normalized_t = session.transpose(&normalized, &[1, 0])?;
                    let tensor_weights = prepare_tensor_weights(session, config, weights, cache)?;
                    let mixed_t = delta_layer_tenferro_cached(
                        session,
                        config,
                        &tensor_weights,
                        &normalized_t,
                        &mask_t,
                        workspace,
                    )?;
                    session.transpose(&mixed_t, &[1, 0])?
                }
                DeltaKernel::HostRecurrent => {
                    let kernel_weights = prepare_kernel_weights(session, config, weights, cache)?;
                    let op = GatedDeltaOp::from_config(config);
                    session.gated_delta(
                        op,
                        &[
                            &normalized,
                            &mask_t,
                            &kernel_weights.qkv,
                            &kernel_weights.z,
                            &kernel_weights.a,
                            &kernel_weights.b,
                            &kernel_weights.conv,
                            &kernel_weights.a_decay,
                            &kernel_weights.dt_bias,
                            &kernel_weights.norm,
                            &kernel_weights.out_proj,
                        ],
                    )?
                }
            },
        };
        let residual = session.add(&hidden, &mixed)?; // (length, hidden)

        let post_norm = cache.col(session, vec![cfg.hidden, 1], &layer.post_norm)?;
        let post_norm = session.reshape(&post_norm, vec![cfg.hidden])?;
        let normalized2 = rms_norm_tenferro(session, &residual, &post_norm, true, cfg.eps as f64)?;

        let gate_w =
            cache.col_major(session, vec![cfg.hidden, cfg.intermediate], &layer.mlp.gate)?;
        let up_w = cache.col_major(session, vec![cfg.hidden, cfg.intermediate], &layer.mlp.up)?;
        let down_w =
            cache.col_major(session, vec![cfg.intermediate, cfg.hidden], &layer.mlp.down)?;
        let gate = linear_tenferro(session, &normalized2, &gate_w)?; // (length, intermediate)
        let up = linear_tenferro(session, &normalized2, &up_w)?;
        let gated = gated_silu_tenferro(session, &gate, &up)?;
        let mlp = linear_tenferro(session, &gated, &down_w)?; // (length, hidden)
        hidden = session.add(&residual, &mlp)?;
    }

    // Final norm on the last position.
    let last = session.slice(
        &hidden,
        tenferro_ad::SliceConfig {
            starts: vec![length - 1, 0],
            limits: vec![length, cfg.hidden],
            strides: vec![1, 1],
        },
    )?; // (1, hidden)
    let final_norm = cache.col(session, vec![cfg.hidden, 1], &weights.final_norm)?;
    let final_norm = session.reshape(&final_norm, vec![cfg.hidden])?;
    let last_normed = rms_norm_tenferro(session, &last, &final_norm, true, cfg.eps as f64)?; // (1, hidden)

    let readout = cache.col_major(session, vec![cfg.hidden, weights.options], &weights.readout)?;
    let logits = linear_tenferro(session, &last_normed, &readout)?; // (1, options)
    extract(session, &logits, 1, weights.options)
}

fn config_error(error: DecisionError) -> tenferro_ad::Error {
    tenferro_ad::Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
        "jeff-infer",
        "config",
        error.to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_ad::EagerRuntime;
    use tenferro_cpu::CpuBackend;

    #[test]
    fn native_linear_f64_preserves_weight_layout() {
        let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
        let output = runtime
            .with_eager_session(|session| {
                // Two rows, three features; columns stored contiguously.
                let x = session.constant_from(tenferro_ad::Tensor::from_vec_col_major(
                    vec![2, 3],
                    vec![1.0f64, 4.0, 2.0, 5.0, 3.0, 6.0],
                )?)?;
                let weight = session.constant_from(tenferro_ad::Tensor::from_vec_col_major(
                    vec![3, 2],
                    vec![1.0f64, 2.0, 3.0, -1.0, 0.0, 1.0],
                )?)?;
                linear_tenferro(session, &x, &weight)
            })
            .unwrap()
            .unwrap();
        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(
            output.value().unwrap().as_slice::<f64>().unwrap(),
            &[14.0, 32.0, 2.0, 2.0]
        );
    }

    struct Lcg(u64);
    impl Lcg {
        fn fill(&mut self, len: usize, lo: f32, hi: f32) -> Vec<f32> {
            (0..len)
                .map(|_| {
                    self.0 = self
                        .0
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let x = ((self.0 >> 40) as f32) / (1u64 << 24) as f32;
                    lo + (hi - lo) * x
                })
                .collect()
        }
    }

    #[test]
    fn projected_q_norm_rope_matches() {
        let cfg = JeffConfig {
            hidden: 8,
            heads: 2,
            head_dim: 4,
            intermediate: 16,
            eps: 1e-5,
        };
        let width = cfg.heads * cfg.head_dim;
        let hd = cfg.head_dim;
        let heads = cfg.heads;
        let mut rng = Lcg(7);
        let q_w = rng.fill(cfg.hidden * width, -0.3, 0.3);
        let q_norm = rng.fill(hd, 0.5, 1.5);
        let length = 3;
        let x = rng.fill(cfg.hidden * length, -1.0, 1.0);

        let projected = linear_host(&q_w, cfg.hidden, width, &x, length);
        let host_normed = rms_heads(&projected, hd, heads, length, &q_norm, cfg.eps);
        let host = rope_partial_host(&host_normed, hd, heads, length, hd, 10000.0);

        let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
        let ten = runtime
            .with_eager_session(|session| -> tenferro_ad::Result<Vec<f32>> {
                let x_t = host_to_tensor(session, cfg.hidden, length, &x)?;
                let q_w_t = host_to_tensor(session, cfg.hidden, width, &q_w)?;
                let proj = linear_col(session, &x_t, &q_w_t)?; // [width,length]
                let shaped = session.reshape(&proj, vec![hd, heads, length])?;
                let heads_first = session.transpose(&shaped, &[1, 2, 0])?; // (heads,length,hd)
                let norm_w = host_to_tensor(session, hd, 1, &q_norm)?;
                let norm_w = session.reshape(&norm_w, vec![hd])?;
                let normed =
                    rms_norm_tenferro(session, &heads_first, &norm_w, true, cfg.eps as f64)?;
                let roped = rope::rope_qwen_partial(session, &normed, 10000.0, hd)?;
                let host_t = session.duplicate_value(&roped)?;
                let values = host_t.as_slice::<f32>()?;
                let mut out = vec![0.0f32; width * length];
                for head in 0..heads {
                    for t in 0..length {
                        for d in 0..hd {
                            out[(head * hd + d) * length + t] =
                                values[head + heads * t + heads * length * d];
                        }
                    }
                }
                Ok(out)
            })
            .unwrap()
            .unwrap();

        let mut max_diff = 0.0f32;
        for (a, b) in host.iter().zip(&ten) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(max_diff <= 1e-4, "projected q norm+rope diff {max_diff}");
    }

    #[test]
    fn full_attention_block_matches() {
        let cfg = JeffConfig {
            hidden: 8,
            heads: 2,
            head_dim: 4,
            intermediate: 16,
            eps: 1e-5,
        };
        let width = cfg.heads * cfg.head_dim;
        let mut rng = Lcg(1);
        let w = FullAttentionWeights {
            q: rng.fill(cfg.hidden * width, -0.3, 0.3),
            gate: rng.fill(cfg.hidden * width, -0.3, 0.3),
            k: rng.fill(cfg.hidden * width, -0.3, 0.3),
            v: rng.fill(cfg.hidden * width, -0.3, 0.3),
            o: rng.fill(width * cfg.hidden, -0.3, 0.3),
            q_norm: rng.fill(cfg.head_dim, 0.5, 1.5),
            k_norm: rng.fill(cfg.head_dim, 0.5, 1.5),
            rope_theta: 10000.0,
            rotary_dim: 2,
        };
        let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
        for length in [1usize, 2, 3, 6] {
            let x = rng.fill(cfg.hidden * length, -1.0, 1.0);
            let mask = vec![1.0f32; length];
            let host = full_attention_host(&w, &cfg, &x, &mask);
            let ten = runtime
                .with_eager_session(|session| {
                    // The forward now works on `(length, hidden)`; the host `x` is
                    // row-major `(hidden, length)`, which is exactly the
                    // column-major `(length, hidden)` buffer.
                    let x_t =
                        session.constant_from_host(tenferro_ad::Tensor::from_vec_col_major(
                            vec![length, cfg.hidden],
                            x.to_vec(),
                        )?)?;
                    let mut cache = TensorCache::new();
                    let out = full_attention_tenferro(
                        session, &mut cache, &w, &cfg, &x_t, &mask, length,
                    )?;
                    let host = session.duplicate_value(&out)?;
                    let values = host.as_slice::<f32>()?;
                    let mut row_major = vec![0.0f32; cfg.hidden * length];
                    for h in 0..cfg.hidden {
                        for l in 0..length {
                            row_major[h * length + l] = values[l + h * length];
                        }
                    }
                    Ok::<Vec<f32>, tenferro_ad::Error>(row_major)
                })
                .unwrap()
                .unwrap();
            let mut max_diff = 0.0f32;
            for (a, b) in host.iter().zip(&ten) {
                max_diff = max_diff.max((a - b).abs());
            }
            eprintln!("length {length}: max diff {max_diff}");
            assert!(
                max_diff <= 1e-3,
                "length {length} block max diff {max_diff}"
            );
        }
    }
}
