//! Jeff typed-decision engine.
//!
//! Phase 5 work in progress. Implemented now (checkpoint-independent):
//!
//! - [`config`]: supported Qwen3.5 `TextConfig` subset and `DecisionConfig`
//! - [`readout`]: temperature-scaled softmax and the choice / noul / score
//!   answer formulas (ported from `extern/JeffClient.jl/src/questions.jl`)
//!
//! Still to come: safetensors loading, embeddings, RMSNorm, partial RoPE, full
//! attention, the reference Gated DeltaNet (`tenferro-gated-delta`), SiLU MLP,
//! and the prepared-token forward pass. Those need a real checkpoint to
//! validate against the reference (`docs/agents/specs/docs/06_ROADMAP.md`
//! Phase 5).

pub mod config;
pub mod readout;

/// Re-export the shared decision types this crate produces.
pub use decision_core;

/// Common imports.
pub mod prelude {
    pub use crate::config::{DecisionConfig, LayerKind, TextConfig};
    pub use crate::readout::{choice_answer, noul_answer, probabilities, score_answer};
}
