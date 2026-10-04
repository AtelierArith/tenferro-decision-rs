//! Laya typed-decision engine.
//!
//! Phase 2 work in progress. Implemented now (checkpoint-independent):
//!
//! - [`config`]: `EncoderConfig` / `AgentConfig` parsing and validation
//! - [`calibration`]: temperature calibration and typed-answer construction
//!   (choice / score / noul), plus the action probability
//! - [`prompt`]: Python-compatible JSON (`py_float` / `py_json_content`),
//!   state/option rendering, and `build_prefix` / `build_sequence` over a
//!   [`prompt::Tokenizer`] trait
//!
//! Still to come: the concrete checkpoint tokenizer (byte-level / Metaspace
//! BPE), safetensors loading, the ModernBERT + decision-head forward pass, and
//! `DecisionEngine` wiring. Those need a real checkpoint (or a captured
//! fixture) to validate against the reference, per
//! `docs/agents/specs/docs/06_ROADMAP.md` Phase 2.

pub mod calibration;
pub mod checkpoint;
pub mod config;
pub mod model;
pub mod prompt;
pub mod tokenizer;

/// Re-export the shared decision types this crate produces.
pub use decision_core;

/// Common imports.
pub mod prelude {
    pub use crate::calibration::{
        action_probability, choice_answer, clamp_temperature, confidence_from_probs, noul_answer,
        round4, score_answer, softmax, temp_bucket, Calibration, LayaDecision, QType,
    };
    pub use crate::config::{AgentConfig, EncoderConfig, LayerKind};
    pub use crate::prompt::{
        build_prefix, build_sequence, py_float, py_json_content, render_criterion, render_options,
        serialize_state, Tokenizer,
    };
}
