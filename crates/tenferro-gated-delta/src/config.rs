//! Gated DeltaNet configuration.

use decision_core::{DecisionError, Result};

/// Which formulation to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algorithm {
    /// Host recurrent reference (no tenferro ops).
    Reference,
    /// Fused host recurrent kernel with a reusable workspace.
    Recurrent,
    /// Tenferro-backed chunked formulation.
    Chunked,
}

/// Layer shape and execution settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GatedDeltaConfig {
    /// Model hidden width.
    pub hidden: usize,
    /// Number of key (query) heads.
    pub key_heads: usize,
    /// Number of value heads.
    pub value_heads: usize,
    /// Key/query head width.
    pub key_dim: usize,
    /// Value head width.
    pub value_dim: usize,
    /// Causal depthwise convolution kernel size.
    pub conv_taps: usize,
    /// RMSNorm epsilon.
    pub eps: f32,
    /// Chunk size for the chunked formulation.
    pub chunk_size: usize,
    /// Formulation to run.
    pub algorithm: Algorithm,
}

impl GatedDeltaConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.key_heads == 0 || self.value_heads == 0 {
            return Err(DecisionError::invalid_field(
                "gated_delta",
                "key_heads and value_heads must be positive",
            ));
        }
        if self.value_heads % self.key_heads != 0 {
            return Err(DecisionError::invalid_field(
                "gated_delta.value_heads",
                "value_heads must be a multiple of key_heads",
            ));
        }
        if self.key_dim == 0 || self.value_dim == 0 {
            return Err(DecisionError::invalid_field(
                "gated_delta",
                "key_dim and value_dim must be positive",
            ));
        }
        if self.hidden == 0 {
            return Err(DecisionError::invalid_field(
                "gated_delta.hidden",
                "hidden must be positive",
            ));
        }
        if self.conv_taps == 0 {
            return Err(DecisionError::invalid_field(
                "gated_delta.conv_taps",
                "conv_taps must be positive",
            ));
        }
        if self.chunk_size == 0 {
            return Err(DecisionError::invalid_field(
                "gated_delta.chunk_size",
                "chunk_size must be positive",
            ));
        }
        Ok(())
    }

    /// The key head serving a value head (consecutive grouping).
    pub fn key_head_for(&self, value_head: usize) -> usize {
        let groups = self.value_heads / self.key_heads;
        (value_head / groups).min(self.key_heads - 1)
    }
}
