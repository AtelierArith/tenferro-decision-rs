//! Jeff typed-decision engine.
//!
//! Phase 5 work in progress. Implemented now (checkpoint-independent):
//!
//! - [`config`]: supported Qwen3.5 `TextConfig` subset and `DecisionConfig`
//! - [`readout`]: temperature-scaled softmax and the choice / noul / score
//!   answer formulas (ported from `extern/JeffClient.jl/src/questions.jl`)
//!
//! Checkpoint-independent model code lives in [`model`], and [`checkpoint`]
//! loads a checkpoint directory (`config.json`, `decision_config.json`,
//! `model.safetensors`, `readout.safetensors`) into prepared [`model::JeffWeights`].
//!
//! Still to come: the prepared-token engine context (`JeffModel`/`JeffContext`
//! per `docs/agents/specs/docs/14_JEFF_INFER_DESIGN.md`).

pub mod checkpoint;
pub mod config;
pub mod engine;
pub mod host_opt;
pub mod model;
pub mod readout;

/// Re-export the shared decision types this crate produces.
pub use decision_core;

/// Common imports.
pub mod prelude {
    pub use crate::config::{DecisionConfig, LayerKind, TextConfig};
    pub use crate::readout::{choice_answer, noul_answer, probabilities, score_answer};
}
