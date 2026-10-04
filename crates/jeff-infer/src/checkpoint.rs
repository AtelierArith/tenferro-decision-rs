//! Checkpoint loading: `config.json`, `decision_config.json`,
//! `model.safetensors`, and `readout.safetensors`.
//!
//! Tensors are read with the shared [`safetensors_io::SafetensorsFile`] reader
//! and transposed/packed to the host layout used by [`crate::model`].
//! Safetensors stores tensors row-major in PyTorch `nn.Linear` layout
//! `(out, in)`. Only `language_model.` tensors from `model.safetensors` are
//! consumed, mirroring `extern/JeffClient.jl/src/native.jl`.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use decision_core::{DecisionError, Result};
use safetensors_io::SafetensorsFile;

use crate::config::{DecisionConfig, LayerKind, TextConfig};
use crate::model::{
    AttentionWeights, FullAttentionWeights, JeffConfig, JeffWeights, LayerWeights, MlpWeights,
};
use tenferro_gated_delta::{Algorithm, GatedDeltaConfig, GatedDeltaWeights};

/// Chunk size used for freshly loaded Gated DeltaNet layers.
pub const DEFAULT_DELTA_CHUNK_SIZE: usize = 64;

/// Prefix applied to every text-model tensor in `model.safetensors`.
const MODEL_PREFIX: &str = "language_model.";

// ------------------------------------------------------------------- loader

/// Checkpoint metadata and prepared model weights.
#[derive(Clone, Debug)]
pub struct JeffCheckpoint {
    /// Parsed `config.json` text configuration.
    pub text: TextConfig,
    /// Parsed `decision_config.json`.
    pub decision: DecisionConfig,
    /// Shared runtime dimensions derived from `text`.
    pub config: JeffConfig,
    /// Prepared model weights in host `(in, out)` layout.
    pub weights: JeffWeights,
}

/// Load a checkpoint directory into prepared [`JeffWeights`].
///
/// Reads `config.json`, `decision_config.json`, `model.safetensors`, and
/// `readout.safetensors`. Tensor names follow
/// `extern/JeffClient.jl/src/native.jl`. The fused `q_proj` output is split per
/// head into `[query(head_dim), gate(head_dim)]` (the layout produced by
/// reshaping `q_proj^T x` to `(2*head_dim, heads, L)` in the reference), and
/// GQA `k`/`v` are expanded to one head per query head with consecutive
/// grouping.
pub fn load_checkpoint(directory: impl AsRef<Path>) -> Result<JeffCheckpoint> {
    let directory = directory.as_ref();
    let text = TextConfig::from_json_str(&read_text(&directory.join("config.json"))?)?;
    let decision =
        DecisionConfig::from_json_str(&read_text(&directory.join("decision_config.json"))?)?;
    text.validate()?;

    let model = SafetensorsFile::open(directory.join("model.safetensors"))?;
    let readout = SafetensorsFile::open(directory.join("readout.safetensors"))?;

    let mut tensors = TensorSet::new(&model);
    let partial = build_weights(&text, &mut tensors)?;
    tensors.finish()?;

    let (options, readout_values) = load_readout(&readout, text.hidden_size)?;
    if decision.max_options > options {
        return Err(DecisionError::invalid_field(
            "decision_config.max_options",
            format!(
                "max_options {} exceeds the {options} readout columns",
                decision.max_options
            ),
        ));
    }

    let inferred = partial.inferred_intermediate(text.hidden_size)?;
    if let Some(declared) = text.intermediate_size {
        if declared != inferred {
            return Err(DecisionError::invalid_field(
                "config.text_config.intermediate_size",
                format!("declared {declared} does not match the {inferred} MLP gate weight rows"),
            ));
        }
    }
    let config = JeffConfig {
        hidden: text.hidden_size,
        heads: text.num_attention_heads,
        head_dim: text.head_dim,
        intermediate: inferred,
        eps: text.rms_norm_eps as f32,
    };
    let weights = JeffWeights {
        embedding: partial.embedding,
        vocab: partial.vocab,
        layers: partial.layers,
        final_norm: partial.final_norm,
        readout: readout_values,
        options,
    };
    weights.validate(&config)?;
    Ok(JeffCheckpoint {
        text,
        decision,
        config,
        weights,
    })
}

/// A partially built model before the readout and derived config are attached.
struct PartialWeights {
    embedding: Vec<f32>,
    vocab: usize,
    layers: Vec<LayerWeights>,
    final_norm: Vec<f32>,
}

impl PartialWeights {
    /// Infer the MLP width from the first layer's gate weight, requiring every
    /// layer to agree.
    fn inferred_intermediate(&self, hidden: usize) -> Result<usize> {
        let mut inferred = None;
        for (index, layer) in self.layers.iter().enumerate() {
            let width = layer.mlp.gate.len() / hidden;
            if width == 0 {
                return Err(DecisionError::invalid_field(
                    "config.text_config.intermediate_size",
                    format!("layer {index} has an empty MLP gate weight"),
                ));
            }
            match inferred {
                Some(expected) if expected != width => {
                    return Err(DecisionError::invalid_field(
                        "config.text_config.intermediate_size",
                        format!(
                            "layer {index} MLP width {width} differs from the earlier {expected}"
                        ),
                    ));
                }
                _ => inferred = Some(width),
            }
        }
        inferred.ok_or_else(|| {
            DecisionError::invalid_field(
                "config.text_config.intermediate_size",
                "cannot infer the MLP intermediate width without a layer",
            )
        })
    }
}

/// Tracks consumed tensors so duplicate or unexpected entries are rejected.
struct TensorSet<'a> {
    file: &'a SafetensorsFile,
    used: HashSet<String>,
}

impl<'a> TensorSet<'a> {
    fn new(file: &'a SafetensorsFile) -> Self {
        Self {
            file,
            used: HashSet::new(),
        }
    }

    /// Mark a tensor used and return its raw `(shape, values)`.
    fn take_any(&mut self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        if !self.used.insert(name.to_string()) {
            return Err(DecisionError::invalid_field(
                "checkpoint.tensor",
                format!("tensor `{name}` is referenced more than once"),
            ));
        }
        self.file.tensor(name)
    }

    /// Mark a tensor used, return its values, and require an exact shape.
    fn take(&mut self, name: &str, want: &[usize]) -> Result<Vec<f32>> {
        let (shape, values) = self.take_any(name)?;
        if shape != want {
            return Err(DecisionError::invalid_field(
                "checkpoint.tensor",
                format!("tensor `{name}` has shape {shape:?}, expected {want:?}"),
            ));
        }
        Ok(values)
    }

    /// Mark a tensor used, returning its values and the validated `(rows, cols)`
    /// of a row-major PyTorch `(out, in)` weight.
    fn take_matrix(&mut self, name: &str, hidden: usize) -> Result<(usize, Vec<f32>)> {
        let (shape, values) = self.take_any(name)?;
        match shape.as_slice() {
            [rows, columns] if *columns == hidden => Ok((*rows, values)),
            _ => Err(DecisionError::invalid_field(
                "checkpoint.tensor",
                format!("tensor `{name}` must be [out, {hidden}], found {shape:?}"),
            )),
        }
    }

    fn finish(self) -> Result<()> {
        for name in self.file.names() {
            if name.starts_with(MODEL_PREFIX) && !self.used.contains(name) {
                return Err(DecisionError::invalid_field(
                    "checkpoint.tensor",
                    format!("unexpected tensor `{name}` with no known destination"),
                ));
            }
        }
        Ok(())
    }
}

fn build_weights(text: &TextConfig, tensors: &mut TensorSet<'_>) -> Result<PartialWeights> {
    let hidden = text.hidden_size;

    let (vocab, embedding) = {
        let name = format!("{MODEL_PREFIX}embed_tokens.weight");
        let (shape, values) = tensors.take_any(&name)?;
        if shape.len() != 2 || shape[1] != hidden {
            return Err(DecisionError::invalid_field(
                "checkpoint.embed_tokens",
                format!("expected shape [vocab, {hidden}], found {shape:?}"),
            ));
        }
        let vocab = shape[0];
        (vocab, transpose(&values, vocab, hidden))
    };

    let mut layers = Vec::with_capacity(text.num_hidden_layers);
    for (index, kind) in text.layer_types.iter().enumerate() {
        let prefix = format!("{MODEL_PREFIX}layers.{index}.");
        let input_norm = tensors.take(&format!("{prefix}input_layernorm.weight"), &[hidden])?;
        let post_norm = tensors.take(
            &format!("{prefix}post_attention_layernorm.weight"),
            &[hidden],
        )?;
        let attention = match kind {
            LayerKind::FullAttention => {
                AttentionWeights::Full(build_full_attention(text, tensors, &prefix, hidden)?)
            }
            LayerKind::LinearAttention => {
                let (weights, config) = build_delta(text, tensors, &prefix, hidden)?;
                AttentionWeights::Delta { weights, config }
            }
        };
        let mlp = build_mlp(tensors, &prefix, hidden)?;
        layers.push(LayerWeights {
            input_norm,
            post_norm,
            attention,
            mlp,
        });
    }

    let final_norm = tensors.take(&format!("{MODEL_PREFIX}norm.weight"), &[hidden])?;

    Ok(PartialWeights {
        embedding,
        vocab,
        layers,
        final_norm,
    })
}

fn build_full_attention(
    text: &TextConfig,
    tensors: &mut TensorSet<'_>,
    prefix: &str,
    hidden: usize,
) -> Result<FullAttentionWeights> {
    let heads = text.num_attention_heads;
    let kv_heads = text.num_key_value_heads;
    let head_dim = text.head_dim;
    let width = heads * head_dim;
    let kv_width = kv_heads * head_dim;
    let groups = heads / kv_heads;

    // The fused q projection emits, per head, `[query(head_dim), gate(head_dim)]`.
    let q_fused = tensors.take(
        &format!("{prefix}self_attn.q_proj.weight"),
        &[2 * width, hidden],
    )?;
    let mut q = vec![0.0f32; hidden * width];
    let mut gate = vec![0.0f32; hidden * width];
    for head in 0..heads {
        for d in 0..head_dim {
            let query_row = head * 2 * head_dim + d;
            let gate_row = query_row + head_dim;
            for input in 0..hidden {
                q[input * width + head * head_dim + d] = q_fused[query_row * hidden + input];
                gate[input * width + head * head_dim + d] = q_fused[gate_row * hidden + input];
            }
        }
    }

    let k_source = tensors.take(
        &format!("{prefix}self_attn.k_proj.weight"),
        &[kv_width, hidden],
    )?;
    let v_source = tensors.take(
        &format!("{prefix}self_attn.v_proj.weight"),
        &[kv_width, hidden],
    )?;
    let mut k = vec![0.0f32; hidden * width];
    let mut v = vec![0.0f32; hidden * width];
    for head in 0..heads {
        let kv_head = (head / groups).min(kv_heads - 1);
        for d in 0..head_dim {
            for input in 0..hidden {
                let source = (kv_head * head_dim + d) * hidden + input;
                let destination = input * width + head * head_dim + d;
                k[destination] = k_source[source];
                v[destination] = v_source[source];
            }
        }
    }

    let o_fused = tensors.take(
        &format!("{prefix}self_attn.o_proj.weight"),
        &[hidden, width],
    )?;
    let o = transpose(&o_fused, hidden, width);

    let q_norm = tensors.take(&format!("{prefix}self_attn.q_norm.weight"), &[head_dim])?;
    let k_norm = tensors.take(&format!("{prefix}self_attn.k_norm.weight"), &[head_dim])?;

    Ok(FullAttentionWeights {
        q,
        gate,
        k,
        v,
        o,
        q_norm,
        k_norm,
        rope_theta: text.rope_theta as f32,
        rotary_dim: text.rotary_dim(),
    })
}

fn build_delta(
    text: &TextConfig,
    tensors: &mut TensorSet<'_>,
    prefix: &str,
    hidden: usize,
) -> Result<(GatedDeltaWeights, GatedDeltaConfig)> {
    let key_heads = text.linear_num_key_heads;
    let value_heads = text.linear_num_value_heads;
    let key_dim = text.linear_key_head_dim;
    let value_dim = text.linear_value_head_dim;
    let key_width = key_dim * key_heads;
    let value_width = value_dim * value_heads;
    let conv_channels = 2 * key_width + value_width;

    let (qkv_rows, qkv) =
        tensors.take_matrix(&format!("{prefix}linear_attn.in_proj_qkv.weight"), hidden)?;
    if qkv_rows != conv_channels {
        return Err(DecisionError::invalid_field(
            "checkpoint.linear_attn.in_proj_qkv",
            format!("expected {conv_channels} output rows, found {qkv_rows}"),
        ));
    }
    let qkv = transpose(&qkv, conv_channels, hidden);

    let (z_rows, z) =
        tensors.take_matrix(&format!("{prefix}linear_attn.in_proj_z.weight"), hidden)?;
    if z_rows != value_width {
        return Err(DecisionError::invalid_field(
            "checkpoint.linear_attn.in_proj_z",
            format!("expected {value_width} output rows, found {z_rows}"),
        ));
    }
    let z = transpose(&z, value_width, hidden);

    let (a_rows, a) =
        tensors.take_matrix(&format!("{prefix}linear_attn.in_proj_a.weight"), hidden)?;
    let (b_rows, b) =
        tensors.take_matrix(&format!("{prefix}linear_attn.in_proj_b.weight"), hidden)?;
    if a_rows != value_heads || b_rows != value_heads {
        return Err(DecisionError::invalid_field(
            "checkpoint.linear_attn.in_proj_a",
            format!("expected {value_heads} output rows for the a/b projections"),
        ));
    }
    let a = transpose(&a, value_heads, hidden);
    let b = transpose(&b, value_heads, hidden);

    let (conv_shape, conv) = tensors.take_any(&format!("{prefix}linear_attn.conv1d.weight"))?;
    let (channels, taps) = match conv_shape.as_slice() {
        [channels, 1, taps] => (*channels, *taps),
        [channels, taps] => (*channels, *taps),
        _ => {
            return Err(DecisionError::invalid_field(
                "checkpoint.conv1d",
                format!(
                    "conv1d weight `{prefix}linear_attn.conv1d.weight` must be [channels, 1, taps], found {conv_shape:?}"
                ),
            ));
        }
    };
    if channels != conv_channels {
        return Err(DecisionError::invalid_field(
            "checkpoint.conv1d",
            format!("conv1d weight has {channels} channels, expected {conv_channels}"),
        ));
    }
    // Host layout is `(taps, channels)`; the file stores `(channels, taps)`.
    let mut conv_host = vec![0.0f32; taps * channels];
    for tap in 0..taps {
        for channel in 0..channels {
            conv_host[tap * channels + channel] = conv[channel * taps + tap];
        }
    }

    let a_log = tensors.take(&format!("{prefix}linear_attn.A_log"), &[value_heads])?;
    let a_decay = a_log.iter().map(|value| -value.exp()).collect();
    let dt_bias = tensors.take(&format!("{prefix}linear_attn.dt_bias"), &[value_heads])?;
    let norm = tensors.take(&format!("{prefix}linear_attn.norm.weight"), &[value_dim])?;

    let (out_rows, out_proj) =
        tensors.take_matrix(&format!("{prefix}linear_attn.out_proj.weight"), value_width)?;
    if out_rows != hidden {
        return Err(DecisionError::invalid_field(
            "checkpoint.linear_attn.out_proj",
            format!("expected {hidden} output rows, found {out_rows}"),
        ));
    }
    let out_proj = transpose(&out_proj, hidden, value_width);

    let weights = GatedDeltaWeights {
        qkv,
        z,
        a,
        b,
        conv: conv_host,
        a_decay,
        dt_bias,
        norm,
        out_proj,
    };
    let config = GatedDeltaConfig {
        hidden,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_taps: taps,
        eps: text.rms_norm_eps as f32,
        chunk_size: DEFAULT_DELTA_CHUNK_SIZE,
        algorithm: Algorithm::Chunked,
    };
    weights.validate(&config)?;
    Ok((weights, config))
}

fn build_mlp(tensors: &mut TensorSet<'_>, prefix: &str, hidden: usize) -> Result<MlpWeights> {
    let (intermediate, gate) =
        tensors.take_matrix(&format!("{prefix}mlp.gate_proj.weight"), hidden)?;
    let gate = transpose(&gate, intermediate, hidden);
    let up = tensors.take(
        &format!("{prefix}mlp.up_proj.weight"),
        &[intermediate, hidden],
    )?;
    let up = transpose(&up, intermediate, hidden);
    let down = tensors.take(
        &format!("{prefix}mlp.down_proj.weight"),
        &[hidden, intermediate],
    )?;
    let down = transpose(&down, hidden, intermediate);
    Ok(MlpWeights { gate, up, down })
}

fn load_readout(file: &SafetensorsFile, hidden: usize) -> Result<(usize, Vec<f32>)> {
    let (shape, values) = file.tensor("weight")?;
    let options = match shape.as_slice() {
        [options, columns] if *columns == hidden => *options,
        _ => {
            return Err(DecisionError::invalid_field(
                "checkpoint.readout",
                format!("readout `weight` must be [options, {hidden}], found {shape:?}"),
            ));
        }
    };
    if options > 255 {
        return Err(DecisionError::invalid_field(
            "checkpoint.readout",
            "readout has more than 255 option columns",
        ));
    }
    Ok((options, transpose(&values, options, hidden)))
}

fn read_text(path: &Path) -> Result<String> {
    fs::read_to_string(path).map_err(|error| io_error(path, error))
}

fn io_error(path: &Path, error: std::io::Error) -> DecisionError {
    DecisionError::Backend {
        message: format!("failed to read `{}`: {error}", path.display()),
        source: Some(Box::new(error)),
    }
}

/// Transpose a row-major `rows x cols` matrix into a row-major `cols x rows`
/// matrix, preserving element order.
fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            out[col * rows + row] = data[row * cols + col];
        }
    }
    out
}
