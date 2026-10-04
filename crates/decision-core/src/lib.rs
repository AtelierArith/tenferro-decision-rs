//! Backend-independent types and the optional [`DecisionEngine`] seam shared by
//! the local engines (`laya-infer`, `jeff-infer`) and the remote client
//! (`jev-client`).
//!
//! This crate deliberately has **no tensor, runtime, or network dependency**
//! (see `docs/agents/specs/docs/16_DECISION_CORE_DESIGN.md`). It owns:
//!
//! - question and answer types with local validation
//! - request/response envelopes and usage metadata
//! - the `Content` value model (allowed JSON-compatible values only)
//! - error categories
//! - the optional `DecisionEngine` trait
//!
//! The wire format of any request is owned by the engine or client that speaks
//! it; this crate only defines the shared, typed surface.

mod answer;
mod content;
mod engine;
mod error;
mod question;
mod response;

pub use answer::{Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer};
pub use content::{Content, MAX_CONTENT_DEPTH};
pub use engine::{DecisionEngine, PreparedState, State};
pub use error::{DecisionError, Result};
pub use question::{
    ChoiceQuestion, NoulCriteria, NoulQuestion, Question, QuestionId, QuestionSet, ScoreQuestion,
    MAX_CHOICE_CANDIDATES, MAX_QUESTIONS, MAX_QUESTION_ID_CHARS, MAX_SCORE_LEVELS,
    MIN_CHOICE_CANDIDATES, MIN_SCORE_LEVELS,
};
pub use response::{SystemOneResponse, Usage};

/// Common imports for engine and client implementations.
pub mod prelude {
    pub use crate::{
        Answer, ChoiceAnswer, ChoiceQuestion, Content, DecisionEngine, DecisionError, NoulAnswer,
        NoulCriteria, NoulQuestion, PreparedState, Question, QuestionId, QuestionSet, ScoreAnswer,
        ScoreQuestion, State, SystemOneResponse, Usage,
    };
}
