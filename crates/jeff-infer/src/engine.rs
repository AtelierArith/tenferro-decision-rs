//! The prepared-token Jeff [`DecisionEngine`].
//!
//! This is the first engine seam for `jeff-infer`: it accepts
//! [`State::Prepared`] tokens only, runs the host reference forward
//! ([`forward_reference`]) once per row, and turns the readout logits into
//! typed answers with [`crate::readout`].
//!
//! Row `i` of the prepared state answers question `i`, so the batch and the
//! [`QuestionSet`] must have the same length. Each question selects its own
//! number of readout columns:
//!
//! - **choice**: one column per candidate id, in criteria order
//! - **noul**: the first two columns (false, true)
//! - **score**: one column per level, in criteria order
//!
//! Natural-language (`Text`) and structured (`Json`) states are not tokenized
//! yet; the initial engine rejects them (`docs/agents/specs/docs/14_JEFF_INFER_DESIGN.md`
//! §9).

use decision_core::{
    Answer, DecisionEngine, DecisionError, PreparedState, Question, QuestionSet, Result, State,
};

use crate::config::DecisionConfig;
use crate::model::{forward_reference, JeffConfig, JeffWeights};
use crate::readout::{choice_answer, noul_answer, score_answer};

/// A prepared-token Jeff engine: dimensions, decision settings, and weights.
#[derive(Clone, Debug)]
pub struct JeffEngine {
    config: JeffConfig,
    decision: DecisionConfig,
    weights: JeffWeights,
}

impl JeffEngine {
    /// Build an engine after validating the weights and decision settings.
    ///
    /// Also requires `decision.max_options` to fit inside the readout, since
    /// every question slices at most that many columns.
    pub fn new(config: JeffConfig, decision: DecisionConfig, weights: JeffWeights) -> Result<Self> {
        weights.validate(&config)?;
        decision.validate()?;
        if decision.max_options > weights.options {
            return Err(DecisionError::invalid_field(
                "decision_config.max_options",
                format!(
                    "max_options {} exceeds the {} readout columns",
                    decision.max_options, weights.options
                ),
            ));
        }
        Ok(Self {
            config,
            decision,
            weights,
        })
    }

    /// The model dimensions.
    pub fn config(&self) -> &JeffConfig {
        &self.config
    }

    /// The decision settings (temperature, option limit).
    pub fn decision(&self) -> &DecisionConfig {
        &self.decision
    }

    /// The model weights.
    pub fn weights(&self) -> &JeffWeights {
        &self.weights
    }

    /// Per-row readout logits `(batch, options)` for a prepared state.
    ///
    /// Rows follow `state.input_ids` order; all readout columns are returned.
    /// This is the raw input to the per-question answer formulas in
    /// [`DecisionEngine::system_one`].
    pub fn logits(&self, state: &PreparedState) -> Result<Vec<Vec<f32>>> {
        state.validate()?;
        state
            .input_ids
            .iter()
            .zip(state.attention_mask.iter())
            .map(|(ids, mask)| self.row_logits(ids, mask))
            .collect()
    }

    /// Run one row and return its readout logits.
    ///
    /// Leading zero-mask positions are trimmed and the row is rebased, matching
    /// the reference's left-padding policy (`native_sequence_start`): RoPE
    /// positions restart at the first active token while interior mask holes
    /// are preserved.
    fn row_logits(&self, ids: &[i64], mask: &[bool]) -> Result<Vec<f32>> {
        let start = mask.iter().position(|active| *active).unwrap_or(0);
        let ids = &ids[start..];
        let mask: Vec<f32> = mask[start..]
            .iter()
            .map(|active| if *active { 1.0 } else { 0.0 })
            .collect();
        forward_reference(&self.config, &self.weights, ids, &mask)
    }

    /// How many readout columns a question consumes.
    fn option_count(question: &Question) -> usize {
        match question {
            Question::Choice(question) => question.criteria.len(),
            Question::Noul(_) => 2,
            Question::Score(question) => question.criteria.len(),
        }
    }

    /// Apply the question's readout formula to the first `count` logits.
    fn answer(&self, question: &Question, logits: &[f32], count: usize) -> Result<Answer> {
        let active: Vec<f64> = logits
            .get(..count)
            .ok_or_else(|| {
                DecisionError::invalid_field(
                    "logits",
                    format!(
                        "readout has {} columns but the question needs {count}",
                        logits.len()
                    ),
                )
            })?
            .iter()
            .map(|value| f64::from(*value))
            .collect();
        let temperature = self.decision.temperature;
        match question {
            Question::Choice(question) => {
                let labels: Vec<String> =
                    question.criteria.iter().map(|(id, _)| id.clone()).collect();
                choice_answer(&labels, &active, temperature).map(Answer::Choice)
            }
            Question::Noul(_) => noul_answer(&active, temperature).map(Answer::Noul),
            Question::Score(question) => {
                score_answer(&question.criteria, &active, temperature).map(Answer::Score)
            }
        }
    }
}

impl DecisionEngine for JeffEngine {
    type Error = DecisionError;

    fn system_one(&mut self, state: &State, questions: &QuestionSet) -> Result<Vec<Answer>> {
        let prepared = match state {
            State::Prepared(prepared) => prepared,
            State::Text(_) => {
                return Err(DecisionError::unsupported(
                    "jeff-infer accepts only prepared token states, not text",
                ))
            }
            State::Json(_) => {
                return Err(DecisionError::unsupported(
                    "jeff-infer accepts only prepared token states, not JSON",
                ))
            }
        };
        prepared.validate()?;
        questions.validate()?;
        if prepared.input_ids.len() != questions.len() {
            return Err(DecisionError::invalid_field(
                "questions",
                format!(
                    "expected one question per prepared row, found {} rows and {} questions",
                    prepared.input_ids.len(),
                    questions.len()
                ),
            ));
        }

        let mut answers = Vec::with_capacity(questions.len());
        for ((_, question), (ids, mask)) in questions.questions().iter().zip(
            prepared
                .input_ids
                .iter()
                .zip(prepared.attention_mask.iter()),
        ) {
            let count = Self::option_count(question);
            if count > self.decision.max_options {
                return Err(DecisionError::invalid_field(
                    "questions",
                    format!(
                        "question needs {count} options but max_options is {}",
                        self.decision.max_options
                    ),
                ));
            }
            let logits = self.row_logits(ids, mask)?;
            answers.push(self.answer(question, &logits, count)?);
        }
        Ok(answers)
    }
}
