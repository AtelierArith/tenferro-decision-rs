//! Laya encoder and agent configuration.
//!
//! Mirrors `EncoderConfig` and the `rl_agent_config.json` validation in
//! `extern/Laya.jl/src/config.jl` and `src/agent.jl`.

use decision_core::{DecisionError, Result};
use serde_json::Value;

/// Attention kind of one encoder layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// Full (global) attention.
    FullAttention,
    /// Sliding-window attention.
    SlidingAttention,
}

impl LayerKind {
    /// Parse the checkpoint's `layer_types` spelling.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "full_attention" => Some(Self::FullAttention),
            "sliding_attention" => Some(Self::SlidingAttention),
            _ => None,
        }
    }

    /// The checkpoint spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FullAttention => "full_attention",
            Self::SlidingAttention => "sliding_attention",
        }
    }
}

/// ModernBERT encoder settings (`encoder/config.json`).
#[derive(Clone, Debug, PartialEq)]
pub struct EncoderConfig {
    /// Token vocabulary size.
    pub vocab_size: usize,
    /// Hidden width.
    pub hidden_size: usize,
    /// MLP intermediate width.
    pub intermediate_size: usize,
    /// Number of encoder layers.
    pub num_hidden_layers: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Model type tag; must be `modernbert`.
    pub model_type: String,
    /// LayerNorm epsilon.
    pub norm_eps: f64,
    /// Whether LayerNorm has a bias.
    pub norm_bias: bool,
    /// Whether attention projections have a bias.
    pub attention_bias: bool,
    /// Whether MLP projections have a bias.
    pub mlp_bias: bool,
    /// Hidden activation name; must be `gelu`.
    pub hidden_activation: String,
    /// Sliding-window size.
    pub local_attention: usize,
    /// Full attention every this many layers.
    pub global_attn_every_n_layers: usize,
    /// RoPE base for full-attention layers.
    pub global_rope_theta: f64,
    /// RoPE base for sliding-attention layers.
    pub local_rope_theta: f64,
    /// Maximum sequence length.
    pub max_position_embeddings: usize,
    /// Per-layer attention kind, one per encoder layer.
    pub layer_types: Vec<LayerKind>,
}

impl EncoderConfig {
    /// Attention head width.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// RoPE base for a layer kind.
    pub fn rope_base(&self, kind: LayerKind) -> f64 {
        match kind {
            LayerKind::FullAttention => self.global_rope_theta,
            LayerKind::SlidingAttention => self.local_rope_theta,
        }
    }

    /// Validate the configuration (mirrors the Julia checks).
    pub fn validate(&self) -> Result<()> {
        if self.model_type != "modernbert" {
            return Err(DecisionError::invalid_field(
                "encoder.model_type",
                format!("expected `modernbert`, found `{}`", self.model_type),
            ));
        }
        if self.hidden_activation != "gelu" {
            return Err(DecisionError::invalid_field(
                "encoder.hidden_activation",
                format!("expected `gelu`, found `{}`", self.hidden_activation),
            ));
        }
        if self.num_attention_heads == 0 || self.hidden_size % self.num_attention_heads != 0 {
            return Err(DecisionError::invalid_field(
                "encoder.hidden_size",
                "hidden_size must be divisible by num_attention_heads",
            ));
        }
        if self.head_dim() % 2 != 0 {
            return Err(DecisionError::invalid_field(
                "encoder.num_attention_heads",
                "the attention head dimension must be even for RoPE",
            ));
        }
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(DecisionError::invalid_field(
                "encoder.layer_types",
                format!(
                    "expected {} entries, found {}",
                    self.num_hidden_layers,
                    self.layer_types.len()
                ),
            ));
        }
        if self.global_attn_every_n_layers == 0 {
            return Err(DecisionError::invalid_field(
                "encoder.global_attn_every_n_layers",
                "must be positive",
            ));
        }
        Ok(())
    }

    /// Parse `encoder/config.json`.
    pub fn from_json_str(text: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(text)
            .map_err(|e| DecisionError::invalid_field("encoder", format!("invalid JSON: {e}")))?;
        Self::from_json(&value)
    }

    /// Parse `encoder/config.json` from a value.
    pub fn from_json(value: &Value) -> Result<Self> {
        let object = value.as_object().ok_or_else(|| {
            DecisionError::invalid_field("encoder", "config must be a JSON object")
        })?;

        let required_usize = |key: &str| -> Result<usize> {
            object
                .get(key)
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| {
                    DecisionError::invalid_field(
                        format!("encoder.{key}"),
                        "expected a positive integer",
                    )
                })
        };

        let num_hidden_layers = required_usize("num_hidden_layers")?;
        let global_attn_every_n_layers = object
            .get("global_attn_every_n_layers")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .unwrap_or(3);

        let layer_types = match object.get("layer_types") {
            Some(Value::Array(items)) => {
                let mut kinds = Vec::with_capacity(items.len());
                for item in items {
                    let name = item.as_str().ok_or_else(|| {
                        DecisionError::invalid_field(
                            "encoder.layer_types",
                            "entries must be strings",
                        )
                    })?;
                    kinds.push(LayerKind::parse(name).ok_or_else(|| {
                        DecisionError::invalid_field(
                            "encoder.layer_types",
                            format!("unknown layer type `{name}`"),
                        )
                    })?);
                }
                kinds
            }
            Some(_) => {
                return Err(DecisionError::invalid_field(
                    "encoder.layer_types",
                    "expected an array",
                ))
            }
            None => default_layer_types(num_hidden_layers, global_attn_every_n_layers),
        };

        let config = Self {
            vocab_size: required_usize("vocab_size")?,
            hidden_size: required_usize("hidden_size")?,
            intermediate_size: required_usize("intermediate_size")?,
            num_hidden_layers,
            num_attention_heads: required_usize("num_attention_heads")?,
            model_type: object
                .get("model_type")
                .and_then(Value::as_str)
                .unwrap_or("modernbert")
                .to_string(),
            norm_eps: number(object, "norm_eps").unwrap_or(1e-5),
            norm_bias: boolean(object, "norm_bias").unwrap_or(false),
            attention_bias: boolean(object, "attention_bias").unwrap_or(false),
            mlp_bias: boolean(object, "mlp_bias").unwrap_or(false),
            hidden_activation: object
                .get("hidden_activation")
                .and_then(Value::as_str)
                .unwrap_or("gelu")
                .to_string(),
            local_attention: object
                .get("local_attention")
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .unwrap_or(128),
            global_attn_every_n_layers,
            global_rope_theta: number(object, "global_rope_theta").unwrap_or(160000.0),
            local_rope_theta: number(object, "local_rope_theta").unwrap_or(10000.0),
            max_position_embeddings: object
                .get("max_position_embeddings")
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .unwrap_or(8192),
            layer_types,
        };
        config.validate()?;
        Ok(config)
    }
}

fn default_layer_types(layers: usize, every: usize) -> Vec<LayerKind> {
    (0..layers)
        .map(|index| {
            if index % every == 0 {
                LayerKind::FullAttention
            } else {
                LayerKind::SlidingAttention
            }
        })
        .collect()
}

/// Agent-level settings (`rl_agent_config.json`).
#[derive(Clone, Debug, PartialEq)]
pub struct AgentConfig {
    /// Number of decision-head transformer layers.
    pub head_layers: usize,
    /// Maximum sequence length.
    pub max_len: usize,
    /// Maximum prefix length (option rendering budget).
    pub head_max_len: usize,
    /// Action names; the action head width.
    pub action_names: Vec<String>,
}

impl AgentConfig {
    /// Number of actions.
    pub fn action_count(&self) -> usize {
        self.action_names.len()
    }

    /// Validate the agent settings against the encoder limits.
    pub fn validate(&self, encoder: &EncoderConfig) -> Result<()> {
        if !(4 < self.head_max_len && self.head_max_len < self.max_len) {
            return Err(DecisionError::invalid_field(
                "rl_agent.head_max_len",
                "expected 4 < head_max_len < max_len",
            ));
        }
        if self.max_len > encoder.max_position_embeddings {
            return Err(DecisionError::invalid_field(
                "rl_agent.max_len",
                "max_len exceeds max_position_embeddings",
            ));
        }
        if self.action_names.is_empty() {
            return Err(DecisionError::invalid_field(
                "rl_agent.act_costs",
                "at least one action is required",
            ));
        }
        Ok(())
    }

    /// Parse `rl_agent_config.json`.
    pub fn from_json_str(text: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(text)
            .map_err(|e| DecisionError::invalid_field("rl_agent", format!("invalid JSON: {e}")))?;
        let object = value.as_object().ok_or_else(|| {
            DecisionError::invalid_field("rl_agent", "config must be a JSON object")
        })?;

        let head_layers = object
            .get("head_layers")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                DecisionError::invalid_field("rl_agent.head_layers", "missing `head_layers`")
            })? as usize;

        let action_names = match object.get("act_costs") {
            Some(Value::Object(map)) => map.keys().cloned().collect(),
            Some(Value::Array(items)) => items
                .iter()
                .map(|item| {
                    item.as_str().map(str::to_string).ok_or_else(|| {
                        DecisionError::invalid_field(
                            "rl_agent.act_costs",
                            "array entries must be strings",
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            Some(Value::Null) | None => Vec::new(),
            Some(_) => {
                return Err(DecisionError::invalid_field(
                    "rl_agent.act_costs",
                    "expected an object or array",
                ))
            }
        };

        Ok(Self {
            head_layers,
            max_len: object
                .get("max_len")
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .unwrap_or(512),
            head_max_len: object
                .get("head_max_len")
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .unwrap_or(192),
            action_names,
        })
    }
}

fn number(object: &serde_json::Map<String, Value>, key: &str) -> Option<f64> {
    object.get(key).and_then(Value::as_f64)
}

fn boolean(object: &serde_json::Map<String, Value>, key: &str) -> Option<bool> {
    object.get(key).and_then(Value::as_bool)
}
