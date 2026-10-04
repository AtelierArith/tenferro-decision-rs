use crate::{Answer, Content, DecisionError, QuestionSet, Result};

/// Prepared token inputs for engines that do not tokenize (the initial
/// `jeff-infer` path).
///
/// The representation is deliberately tensor-free: `decision-core` must not
/// depend on tenferro (`docs/agents/specs/docs/16_DECISION_CORE_DESIGN.md` §5).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedState {
    /// Token ids, one row per batch element.
    pub input_ids: Vec<Vec<i64>>,
    /// Active-token mask, shape-matched to `input_ids`.
    pub attention_mask: Vec<Vec<bool>>,
}

impl PreparedState {
    /// Validate batch consistency: matching shapes, non-empty batch, and at
    /// least one active token per row.
    pub fn validate(&self) -> Result<()> {
        if self.input_ids.is_empty() {
            return Err(DecisionError::invalid_field(
                "state.input_ids",
                "at least one batch element is required",
            ));
        }
        if self.input_ids.len() != self.attention_mask.len() {
            return Err(DecisionError::invalid_field(
                "state.attention_mask",
                "must have one mask row per input row",
            ));
        }
        for (index, (ids, mask)) in self
            .input_ids
            .iter()
            .zip(self.attention_mask.iter())
            .enumerate()
        {
            if ids.len() != mask.len() {
                return Err(DecisionError::invalid_field(
                    format!("state.attention_mask[{index}]"),
                    "mask row length must match the input row",
                ));
            }
            if !mask.iter().any(|active| *active) {
                return Err(DecisionError::invalid_field(
                    format!("state.attention_mask[{index}]"),
                    "each batch element must have at least one active token",
                ));
            }
        }
        Ok(())
    }
}

/// The input to a decision run.
///
/// The variants reflect the three input shapes in the package: natural
/// language (Laya), prepared tokens (Jeff), and structured JSON (Jev). Which
/// variant an engine accepts is engine-specific
/// (`docs/agents/specs/docs/10_DECISION_ABSTRACTION.md` §4).
#[derive(Clone, Debug, PartialEq)]
pub enum State {
    /// Natural-language state.
    Text(String),
    /// Structured, JSON-compatible state.
    Json(Content),
    /// Prepared token ids and mask.
    Prepared(PreparedState),
}

impl State {
    /// Validate the state.
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Text(text) if text.is_empty() => Err(DecisionError::invalid_field(
                "state",
                "text state must not be empty",
            )),
            Self::Text(_) => Ok(()),
            Self::Json(content) => content.validate(),
            Self::Prepared(prepared) => prepared.validate(),
        }
    }
}

/// The adapter seam shared by every answering engine.
///
/// This trait is intentionally thin and is introduced only after the Laya
/// engine is stable (`docs/agents/specs/docs/10_DECISION_ABSTRACTION.md` §4).
/// Answers are returned in [`QuestionSet`] order, so their identifiers are
/// implied by the request. Implementations own their error type; the engines
/// are free to use [`DecisionError`] directly.
pub trait DecisionEngine {
    /// Engine-specific failure type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Answer every question against `state`.
    fn system_one(
        &mut self,
        state: &State,
        questions: &QuestionSet,
    ) -> std::result::Result<Vec<Answer>, Self::Error>;
}
