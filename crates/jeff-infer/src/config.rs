//! Jeff configuration (`config.json` and `decision_config.json`).
//!
//! Mirrors the validation in `extern/JeffClient.jl/src/native.jl`.

use decision_core::{DecisionError, Result};
use serde_json::Value;

/// One text-model layer kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// Full (softmax) attention.
    FullAttention,
    /// Gated DeltaNet linear attention.
    LinearAttention,
}

impl LayerKind {
    /// Parse the checkpoint spelling.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "full_attention" => Some(Self::FullAttention),
            "linear_attention" => Some(Self::LinearAttention),
            _ => None,
        }
    }
}

/// The supported Qwen3.5 text configuration subset.
#[derive(Clone, Debug, PartialEq)]
pub struct TextConfig {
    /// Hidden width.
    pub hidden_size: usize,
    /// Attention head width.
    pub head_dim: usize,
    /// Number of query heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads (GQA).
    pub num_key_value_heads: usize,
    /// Number of layers.
    pub num_hidden_layers: usize,
    /// MLP intermediate width when the checkpoint records it.
    ///
    /// Optional so the loader can infer it from the MLP gate weight shape when
    /// a minimal `config.json` omits it.
    pub intermediate_size: Option<usize>,
    /// Number of key heads in the linear-attention layers.
    pub linear_num_key_heads: usize,
    /// Number of value heads in the linear-attention layers.
    pub linear_num_value_heads: usize,
    /// Key head width in the linear-attention layers.
    pub linear_key_head_dim: usize,
    /// Value head width in the linear-attention layers.
    pub linear_value_head_dim: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f64,
    /// RoPE base.
    pub rope_theta: f64,
    /// Fraction of each head rotated.
    pub partial_rotary_factor: f64,
    /// Per-layer kinds.
    pub layer_types: Vec<LayerKind>,
}

impl TextConfig {
    /// Number of rotated channels per head.
    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f64 * self.partial_rotary_factor) as usize
    }

    /// Validate the subset.
    pub fn validate(&self) -> Result<()> {
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(DecisionError::invalid_field(
                "text_config.num_attention_heads",
                "must be divisible by num_key_value_heads",
            ));
        }
        if self.linear_num_value_heads % self.linear_num_key_heads != 0 {
            return Err(DecisionError::invalid_field(
                "text_config.linear_num_value_heads",
                "must be divisible by linear_num_key_heads",
            ));
        }
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(DecisionError::invalid_field(
                "text_config.layer_types",
                "one layer type per layer is required",
            ));
        }
        Ok(())
    }

    /// Parse the `text_config` object of `config.json`.
    pub fn from_json_str(text: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(text)
            .map_err(|e| DecisionError::invalid_field("config", format!("invalid JSON: {e}")))?;
        Self::from_json(&value)
    }

    /// Parse from a `config.json` value (top level or `text_config`).
    pub fn from_json(value: &Value) -> Result<Self> {
        let object = value.as_object().ok_or_else(|| {
            DecisionError::invalid_field("config", "config must be a JSON object")
        })?;

        if let Some(model_type) = object.get("model_type").and_then(Value::as_str) {
            if model_type != "qwen3_5" {
                return Err(DecisionError::invalid_field(
                    "config.model_type",
                    format!("expected `qwen3_5`, found `{model_type}`"),
                ));
            }
        }
        if let Some(attention_bias) = object.get("attention_bias").and_then(Value::as_bool) {
            if attention_bias {
                return Err(DecisionError::unsupported(
                    "attention_bias is not supported",
                ));
            }
        }
        if let Some(hidden_act) = object.get("hidden_act").and_then(Value::as_str) {
            if hidden_act != "silu" {
                return Err(DecisionError::invalid_field(
                    "config.hidden_act",
                    format!("expected `silu`, found `{hidden_act}`"),
                ));
            }
        }

        let text = match object.get("text_config") {
            Some(Value::Object(map)) => Value::Object(map.clone()),
            _ => value.clone(),
        };
        let text = text.as_object().ok_or_else(|| {
            DecisionError::invalid_field("text_config", "text_config must be an object")
        })?;

        let required = |key: &str| -> Result<usize> {
            text.get(key)
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| {
                    DecisionError::invalid_field(
                        format!("text_config.{key}"),
                        "expected a positive integer",
                    )
                })
        };

        let num_hidden_layers = required("num_hidden_layers")?;
        let layer_types = match text.get("layer_types") {
            Some(Value::Array(items)) => items
                .iter()
                .map(|item| {
                    let name = item.as_str().ok_or_else(|| {
                        DecisionError::invalid_field(
                            "text_config.layer_types",
                            "entries must be strings",
                        )
                    })?;
                    LayerKind::parse(name).ok_or_else(|| {
                        DecisionError::invalid_field(
                            "text_config.layer_types",
                            format!("unknown layer type `{name}`"),
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            _ => vec![LayerKind::FullAttention; num_hidden_layers],
        };

        let rope_theta = text
            .get("rope_parameters")
            .and_then(Value::as_object)
            .and_then(|params| params.get("rope_theta"))
            .and_then(Value::as_f64)
            .unwrap_or(1_000_000.0);
        let partial_rotary_factor = text
            .get("rope_parameters")
            .and_then(Value::as_object)
            .and_then(|params| params.get("partial_rotary_factor"))
            .and_then(Value::as_f64)
            .unwrap_or(1.0);

        let config = Self {
            hidden_size: required("hidden_size")?,
            head_dim: required("head_dim")?,
            num_attention_heads: required("num_attention_heads")?,
            num_key_value_heads: required("num_key_value_heads")?,
            num_hidden_layers,
            intermediate_size: text
                .get("intermediate_size")
                .and_then(Value::as_u64)
                .map(|v| v as usize),
            linear_num_key_heads: required("linear_num_key_heads")?,
            linear_num_value_heads: required("linear_num_value_heads")?,
            linear_key_head_dim: required("linear_key_head_dim")?,
            linear_value_head_dim: required("linear_value_head_dim")?,
            rms_norm_eps: text
                .get("rms_norm_eps")
                .and_then(Value::as_f64)
                .unwrap_or(1e-6),
            rope_theta,
            partial_rotary_factor,
            layer_types,
        };
        config.validate()?;
        Ok(config)
    }
}

/// `decision_config.json` settings.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionConfig {
    /// Export format version; must be `1`.
    pub format_version: u64,
    /// Fitted softmax temperature.
    pub temperature: f64,
    /// Maximum number of options (columns).
    pub max_options: usize,
}

impl DecisionConfig {
    /// Validate the settings.
    pub fn validate(&self) -> Result<()> {
        if self.format_version != 1 {
            return Err(DecisionError::unsupported(format!(
                "decision format_version {} is not supported",
                self.format_version
            )));
        }
        if !(self.temperature.is_finite() && self.temperature > 0.0) {
            return Err(DecisionError::invalid_field(
                "decision_config.temperature",
                "temperature must be finite and positive",
            ));
        }
        if !(1..=255).contains(&self.max_options) {
            return Err(DecisionError::invalid_field(
                "decision_config.max_options",
                "max_options must be between 1 and 255",
            ));
        }
        Ok(())
    }

    /// Parse `decision_config.json`.
    pub fn from_json_str(text: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(text).map_err(|e| {
            DecisionError::invalid_field("decision_config", format!("invalid JSON: {e}"))
        })?;
        Self::from_json(&value)
    }

    /// Parse from a value.
    pub fn from_json(value: &Value) -> Result<Self> {
        let object = value.as_object().ok_or_else(|| {
            DecisionError::invalid_field("decision_config", "must be a JSON object")
        })?;
        let config = Self {
            format_version: object
                .get("format_version")
                .and_then(Value::as_u64)
                .unwrap_or(1),
            temperature: object
                .get("temperature")
                .and_then(Value::as_f64)
                .ok_or_else(|| {
                    DecisionError::invalid_field(
                        "decision_config.temperature",
                        "missing temperature",
                    )
                })?,
            max_options: object
                .get("max_options")
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| {
                    DecisionError::invalid_field(
                        "decision_config.max_options",
                        "missing max_options",
                    )
                })?,
        };
        config.validate()?;
        Ok(config)
    }
}
