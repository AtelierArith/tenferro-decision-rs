//! The natural-language Laya [`DecisionEngine`].
//!
//! Wires the concrete pieces together: the [`BpeTokenizer`] from a checkpoint's
//! `tokenizer/` assets, the [`build_sequence`] prompt (which yields the
//! `[MASK]` marker positions), the ModernBERT + decision-head forward, and the
//! Laya calibration and answer construction.
//!
//! Each question is answered as its own batch row (the prompt, marker count and
//! question type differ per question), which keeps the forward free of
//! cross-question padding and is bit-identical to the reference for a batch of
//! one. [`State::Prepared`] is rejected: Laya is text/JSON only.

use decision_core::{Answer, DecisionEngine, DecisionError, Question, QuestionSet, Result, State};

use std::path::Path;

use crate::calibration::{
    action_probability, choice_answer, noul_answer, score_answer, Calibration, LayaDecision, QType,
};
use crate::checkpoint::LayaCheckpoint;
use crate::config::{AgentConfig, EncoderConfig};
use crate::model::{forward_reference, LayaWeights};
use crate::prompt::build_sequence;
use crate::tokenizer::BpeTokenizer;

/// A fully loaded Laya engine.
#[derive(Debug)]
pub struct LayaEngine {
    encoder: EncoderConfig,
    agent: AgentConfig,
    weights: LayaWeights,
    tokenizer: BpeTokenizer,
    calibration: Calibration,
}

impl LayaEngine {
    /// Build an engine from validated parts.
    pub fn new(
        encoder: EncoderConfig,
        agent: AgentConfig,
        weights: LayaWeights,
        tokenizer: BpeTokenizer,
        calibration: Calibration,
    ) -> Result<Self> {
        agent.validate(&encoder)?;
        weights.validate(&encoder, &agent)?;
        Ok(Self {
            encoder,
            agent,
            weights,
            tokenizer,
            calibration,
        })
    }

    /// Build an engine from a loaded checkpoint plus tokenizer and calibration.
    pub fn from_checkpoint(
        checkpoint: LayaCheckpoint,
        tokenizer: BpeTokenizer,
        calibration: Calibration,
    ) -> Result<Self> {
        Self::new(
            checkpoint.encoder,
            checkpoint.agent,
            checkpoint.weights,
            tokenizer,
            calibration,
        )
    }

    /// Load a complete Laya checkpoint directory: encoder/agent configs and
    /// weights (`load_checkpoint`), the `tokenizer/` assets, and the calibration
    /// from `rl_agent_config.json`.
    pub fn load(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let checkpoint = crate::checkpoint::load_checkpoint(directory)?;
        let calibration_text = std::fs::read_to_string(directory.join("rl_agent_config.json"))
            .map_err(|error| DecisionError::Backend {
                message: format!("failed to read rl_agent_config.json: {error}"),
                source: Some(Box::new(error)),
            })?;
        let calibration = Calibration::from_json_str(&calibration_text)?;
        let tokenizer = BpeTokenizer::from_directory(directory.join("tokenizer"))?;
        Self::from_checkpoint(checkpoint, tokenizer, calibration)
    }

    /// The encoder configuration.
    pub fn encoder(&self) -> &EncoderConfig {
        &self.encoder
    }

    /// The agent (decision-head) configuration.
    pub fn agent(&self) -> &AgentConfig {
        &self.agent
    }

    /// The model weights.
    pub fn weights(&self) -> &LayaWeights {
        &self.weights
    }

    /// The tokenizer.
    pub fn tokenizer(&self) -> &BpeTokenizer {
        &self.tokenizer
    }

    /// The calibration.
    pub fn calibration(&self) -> &Calibration {
        &self.calibration
    }

    /// Answer every question and report the action probability.
    pub fn decide(&self, state: &State, questions: &QuestionSet) -> Result<Vec<LayaDecision>> {
        state.validate()?;
        questions.validate()?;
        questions
            .questions()
            .iter()
            .map(|(_, question)| self.decide_one(state, question))
            .collect()
    }

    /// Answer one question for the whole state.
    fn decide_one(&self, state: &State, question: &Question) -> Result<LayaDecision> {
        let (ids, markers) = build_sequence(
            &self.tokenizer,
            state,
            question,
            self.agent.max_len,
            self.agent.head_max_len,
        )?;
        if markers.is_empty() {
            return Err(DecisionError::invalid_field(
                "laya.markers",
                "the prompt produced no marker positions",
            ));
        }
        let used = markers.len();
        // The runtime pads to at least two marker slots for one-option choices.
        let slots = used.max(2);
        let mut marker_pos = vec![0_i64; slots];
        let mut marker_mask = vec![false; slots];
        for (slot, position) in markers.iter().enumerate() {
            marker_pos[slot] = *position as i64;
            marker_mask[slot] = true;
        }
        let mask = vec![true; ids.len()];
        let qtype = question_type(question);
        let (logits, action) = forward_reference(
            &self.encoder,
            &self.agent,
            &self.weights,
            &ids,
            &mask,
            &marker_pos,
            &marker_mask,
            &[qtype.index() as i64],
        )?;

        let active: Vec<f64> = logits
            .get(..used)
            .ok_or_else(|| {
                DecisionError::invalid_field(
                    "laya.logits",
                    format!(
                        "forward returned {} slots but the prompt needs {used}",
                        logits.len()
                    ),
                )
            })?
            .iter()
            .map(|value| f64::from(*value))
            .collect();
        let probabilities = self.calibration.probabilities(qtype, used, &active);
        let answer = match question {
            Question::Choice(question) => {
                let labels: Vec<String> =
                    question.criteria.iter().map(|(id, _)| id.clone()).collect();
                Answer::Choice(choice_answer(&labels, &probabilities))
            }
            Question::Score(question) => {
                Answer::Score(score_answer(&question.criteria, &probabilities))
            }
            Question::Noul(_) => Answer::Noul(noul_answer(&probabilities)),
        };
        let action_logits: Vec<f64> = action.iter().map(|value| f64::from(*value)).collect();
        Ok(LayaDecision {
            answer,
            action_probability: action_probability(&action_logits),
        })
    }
}

/// The model's numeric question type for a [`Question`].
fn question_type(question: &Question) -> QType {
    match question {
        Question::Choice(_) => QType::Choice,
        Question::Score(_) => QType::Score,
        Question::Noul(_) => QType::Noul,
    }
}

impl DecisionEngine for LayaEngine {
    type Error = DecisionError;

    fn system_one(&mut self, state: &State, questions: &QuestionSet) -> Result<Vec<Answer>> {
        match state {
            State::Prepared(_) => Err(DecisionError::unsupported(
                "laya-infer accepts text or JSON states, not prepared tokens",
            )),
            State::Text(_) | State::Json(_) => Ok(self
                .decide(state, questions)?
                .into_iter()
                .map(|decision| decision.answer)
                .collect()),
        }
    }
}
