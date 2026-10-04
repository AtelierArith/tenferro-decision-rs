//! Laya checkpoint loading.
//!
//! Reads `encoder/config.json`, `rl_agent_config.json` and `model.safetensors`
//! from a Laya checkpoint directory and prepares the host-layout weights used by
//! [`crate::model`]. The tensor names, the name sanitization and the strict
//! consumption rules mirror the `DecisionModel` / `WeightReader` construction in
//! `extern/Laya.jl/src/weights.jl`.
//!
//! ## Layout
//!
//! [`crate::model::LinearWeights::weight`] is column-major `(in, out)` flattened
//! as `weight[i + in * o]`. Safetensors stores PyTorch `nn.Linear` weights
//! row-major as `(out, in)`, flattened as `file[o * in + i]`. Those indices are
//! equal (`i + in * o == o * in + i`), so **linears pass through unchanged**.
//! The same identity holds for `tok_embeddings` (file `(vocab, d)` row-major is
//! byte-identical to host `(hidden, vocab)` column-major) and `type_emb` (file
//! `(3, d)` is byte-identical to host `(hidden, 3)`). Biases are `(out,)` and
//! are copied unchanged.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use decision_core::{DecisionError, Result};
use safetensors_io::SafetensorsFile;

use crate::config::{AgentConfig, EncoderConfig};
use crate::model::{
    EncoderLayerWeights, HeadLayerWeights, LayaWeights, LayerNormWeights, LinearWeights,
    ModernBertWeights,
};

/// The action-head hidden width used by the Laya reference (`weights.jl`).
const ACTION_HIDDEN: usize = 256;

/// Checkpoint metadata and prepared model weights.
#[derive(Clone, Debug)]
pub struct LayaCheckpoint {
    /// Parsed `encoder/config.json`.
    pub encoder: EncoderConfig,
    /// Parsed `rl_agent_config.json`.
    pub agent: AgentConfig,
    /// Prepared model weights in host `(in, out)` layout.
    pub weights: LayaWeights,
}

/// Load a Laya checkpoint directory into a [`LayaCheckpoint`].
///
/// Reads `encoder/config.json`, `rl_agent_config.json` and `model.safetensors`.
/// Tensor names are normalized exactly like `sanitize_weights` in `weights.jl`:
/// the fused attention parameter spellings `.in_proj_weight` / `.in_proj_bias`
/// become `.in_proj.weight` / `.in_proj.bias`, and top-level `scorer.<x>` /
/// `act_head.<x>` become `scorer.layers.<x>` / `act_head.layers.<x>`. Both the
/// upstream PyTorch spellings and the already-converted MLX `layers.` forms are
/// accepted. An optional `temperature` buffer is accepted and ignored. Every
/// other tensor must be consumed exactly once, and the loaded weights are
/// validated against both configs before returning.
pub fn load_checkpoint(directory: impl AsRef<Path>) -> Result<LayaCheckpoint> {
    let directory = directory.as_ref();
    let encoder =
        EncoderConfig::from_json_str(&read_text(&directory.join("encoder").join("config.json"))?)?;
    let agent = AgentConfig::from_json_str(&read_text(&directory.join("rl_agent_config.json"))?)?;
    // The reference validates the agent limits against the encoder limits.
    agent.validate(&encoder)?;

    let file = SafetensorsFile::open(directory.join("model.safetensors"))?;
    let mut tensors = TensorSet::new(&file)?;
    let weights = build_weights(&encoder, &agent, &mut tensors)?;
    tensors.finish()?;
    weights.validate(&encoder, &agent)?;
    Ok(LayaCheckpoint {
        encoder,
        agent,
        weights,
    })
}

/// Normalize a checkpoint parameter name to the canonical `weights.jl` spelling.
fn normalize_name(name: &str) -> String {
    let mut name = name
        .replace(".in_proj_weight", ".in_proj.weight")
        .replace(".in_proj_bias", ".in_proj.bias");
    for prefix in ["scorer", "act_head"] {
        let with_dot = format!("{prefix}.");
        let nested = format!("{prefix}.layers.");
        if name.starts_with(&with_dot) && !name.starts_with(&nested) {
            name = format!("{prefix}.layers.{}", &name[with_dot.len()..]);
        }
    }
    name
}

/// A safetensors reader keyed by normalized parameter names.
struct TensorSet<'a> {
    file: &'a SafetensorsFile,
    /// Normalized name -> the original name in the file.
    originals: HashMap<String, String>,
    /// Original names already consumed.
    used: HashSet<String>,
}

impl<'a> TensorSet<'a> {
    /// Build the name map, rejecting collisions introduced by normalization.
    fn new(file: &'a SafetensorsFile) -> Result<Self> {
        let mut originals = HashMap::new();
        for name in file.names() {
            let normalized = normalize_name(name);
            if let Some(previous) = originals.insert(normalized.clone(), name.to_string()) {
                return Err(DecisionError::invalid_field(
                    "checkpoint.tensor",
                    format!(
                        "duplicate parameter `{normalized}` after conversion (from `{previous}` and `{name}`)"
                    ),
                ));
            }
        }
        Ok(Self {
            file,
            originals,
            used: HashSet::new(),
        })
    }

    /// Mark a tensor used and return its raw `(shape, values)`.
    fn take_any(&mut self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let original = self.originals.get(name).cloned().ok_or_else(|| {
            DecisionError::invalid_field(
                "checkpoint.tensor",
                format!("missing checkpoint parameter `{name}`"),
            )
        })?;
        if !self.used.insert(original.clone()) {
            return Err(DecisionError::invalid_field(
                "checkpoint.tensor",
                format!("parameter `{name}` is referenced more than once"),
            ));
        }
        self.file.tensor(&original)
    }

    /// Mark a tensor used and require an exact shape.
    fn take(&mut self, name: &str, want: &[usize]) -> Result<Vec<f32>> {
        let (shape, values) = self.take_any(name)?;
        if shape != want {
            return Err(DecisionError::invalid_field(
                "checkpoint.tensor",
                format!("parameter `{name}` has shape {shape:?}, expected {want:?}"),
            ));
        }
        Ok(values)
    }

    /// Reject any tensor that was not consumed, except the ignored buffer.
    fn finish(self) -> Result<()> {
        for name in self.file.names() {
            if name != "temperature" && !self.used.contains(name) {
                return Err(DecisionError::invalid_field(
                    "checkpoint.tensor",
                    format!("unexpected checkpoint parameter `{name}`"),
                ));
            }
        }
        Ok(())
    }
}

/// Load a dense layer stored as a PyTorch `(out, in)` weight and optional `(out,)`
/// bias.
fn load_linear(
    tensors: &mut TensorSet<'_>,
    name: &str,
    in_dim: usize,
    out_dim: usize,
    with_bias: bool,
) -> Result<LinearWeights> {
    // The file's row-major `(out, in)` array is the host's column-major `(in, out)`
    // array without any transpose; see the module note.
    let weight = tensors.take(&format!("{name}.weight"), &[out_dim, in_dim])?;
    let bias = if with_bias {
        Some(tensors.take(&format!("{name}.bias"), &[out_dim])?)
    } else {
        None
    };
    Ok(LinearWeights { weight, bias })
}

/// Load a LayerNorm `(d,)` scale and optional `(d,)` bias.
fn load_norm(
    tensors: &mut TensorSet<'_>,
    name: &str,
    dim: usize,
    with_bias: bool,
) -> Result<LayerNormWeights> {
    let weight = tensors.take(&format!("{name}.weight"), &[dim])?;
    let bias = if with_bias {
        Some(tensors.take(&format!("{name}.bias"), &[dim])?)
    } else {
        None
    };
    Ok(LayerNormWeights { weight, bias })
}

/// Build the full model weights, consuming exactly the expected tensors.
fn build_weights(
    encoder: &EncoderConfig,
    agent: &AgentConfig,
    tensors: &mut TensorSet<'_>,
) -> Result<LayaWeights> {
    let d = encoder.hidden_size;
    let intermediate = encoder.intermediate_size;

    let tok_embeddings = tensors.take(
        "encoder.embeddings.tok_embeddings.weight",
        &[encoder.vocab_size, d],
    )?;
    let embed_norm = load_norm(tensors, "encoder.embeddings.norm", d, encoder.norm_bias)?;

    let mut layers = Vec::with_capacity(encoder.num_hidden_layers);
    for index in 0..encoder.num_hidden_layers {
        let prefix = format!("encoder.layers.{index}");
        let kind = encoder.layer_types.get(index).copied().ok_or_else(|| {
            DecisionError::invalid_field(
                "encoder.layer_types",
                format!("missing entry for layer {index}"),
            )
        })?;
        // The first layer's attention normalization is the identity.
        let attn_norm = if index == 0 {
            None
        } else {
            Some(load_norm(
                tensors,
                &format!("{prefix}.attn_norm"),
                d,
                encoder.norm_bias,
            )?)
        };
        layers.push(EncoderLayerWeights {
            kind,
            attn_norm,
            wqkv: load_linear(
                tensors,
                &format!("{prefix}.attn.Wqkv"),
                d,
                3 * d,
                encoder.attention_bias,
            )?,
            wo: load_linear(
                tensors,
                &format!("{prefix}.attn.Wo"),
                d,
                d,
                encoder.attention_bias,
            )?,
            num_heads: encoder.num_attention_heads,
            rope_base: encoder.rope_base(kind),
            mlp_norm: load_norm(tensors, &format!("{prefix}.mlp_norm"), d, encoder.norm_bias)?,
            wi: load_linear(
                tensors,
                &format!("{prefix}.mlp.Wi"),
                d,
                2 * intermediate,
                encoder.mlp_bias,
            )?,
            wo_mlp: load_linear(
                tensors,
                &format!("{prefix}.mlp.Wo"),
                intermediate,
                d,
                encoder.mlp_bias,
            )?,
        });
    }

    let final_norm = load_norm(tensors, "encoder.final_norm", d, encoder.norm_bias)?;
    let encoder_weights = ModernBertWeights {
        tok_embeddings,
        embed_norm,
        layers,
        final_norm,
    };

    // The decision head uses one attention head per 64 hidden units (at least one),
    // matching `nheads = max(1, d ÷ 64)` in `weights.jl`.
    let head_heads = (d / 64).max(1);
    let mut head = Vec::with_capacity(agent.head_layers);
    for index in 0..agent.head_layers {
        let prefix = format!("head.layers.{index}");
        head.push(HeadLayerWeights {
            num_heads: head_heads,
            norm1: load_norm(tensors, &format!("{prefix}.norm1"), d, true)?,
            in_proj: load_linear(
                tensors,
                &format!("{prefix}.self_attn.in_proj"),
                d,
                3 * d,
                true,
            )?,
            out_proj: load_linear(tensors, &format!("{prefix}.self_attn.out_proj"), d, d, true)?,
            norm2: load_norm(tensors, &format!("{prefix}.norm2"), d, true)?,
            linear1: load_linear(tensors, &format!("{prefix}.linear1"), d, 4 * d, true)?,
            linear2: load_linear(tensors, &format!("{prefix}.linear2"), 4 * d, d, true)?,
        });
    }

    let type_emb = tensors.take("type_emb.weight", &[3, d])?;
    let scorer_norm = load_norm(tensors, "scorer.layers.0", d, true)?;
    let scorer1 = load_linear(tensors, "scorer.layers.1", d, d, true)?;
    let scorer2 = load_linear(tensors, "scorer.layers.3", d, 1, true)?;
    let act1 = load_linear(tensors, "act_head.layers.0", d + 4, ACTION_HIDDEN, true)?;
    let act2 = load_linear(
        tensors,
        "act_head.layers.2",
        ACTION_HIDDEN,
        agent.action_count(),
        true,
    )?;

    Ok(LayaWeights {
        encoder: encoder_weights,
        head,
        type_emb,
        scorer_norm,
        scorer1,
        scorer2,
        act1,
        act2,
    })
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
