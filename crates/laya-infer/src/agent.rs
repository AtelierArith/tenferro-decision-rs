//! The natural-language Laya [`DecisionEngine`].
//!
//! Wires the concrete pieces together: the [`BpeTokenizer`] from a checkpoint's
//! `tokenizer/` assets, the [`build_sequence`] prompt (which yields the
//! `[MASK]` marker positions), the ModernBERT + decision-head forward, and the
//! Laya calibration and answer construction.
//!
//! Questions are collated into bounded padded batches, matching the Julia runtime.
//! Marker masks preserve each row's option count. [`State::Prepared`] is
//! rejected: Laya is text/JSON only.

use decision_core::{
    Answer, DecisionEngine, DecisionError, Question, QuestionId, QuestionSet, Result, State,
};

use std::path::Path;
use std::sync::Arc;

use tenferro_ad::EagerRuntime;
pub use tenferro_ext::Device;

use crate::calibration::{
    Calibration, LayaDecision, QType, action_probability, choice_answer, noul_answer, score_answer,
};
use crate::checkpoint::LayaCheckpoint;
use crate::config::{AgentConfig, EncoderConfig};
use crate::model::{LayaWeights, TensorCache, forward_tenferro_cached};
use crate::prompt::{Tokenizer, build_sequence};
use crate::tokenizer::BpeTokenizer;

/// A fully loaded Laya engine.
#[derive(Debug)]
pub struct LayaEngine {
    encoder: EncoderConfig,
    agent: AgentConfig,
    weights: LayaWeights,
    tokenizer: BpeTokenizer,
    calibration: Calibration,
    batch_size: usize,
    /// The device of `runtime`.
    device: Device,
    /// The tenferro runtime the production forward runs on.
    runtime: Arc<EagerRuntime>,
    /// Weight tensors cached across forwards (see [`TensorCache`]).
    cache: TensorCache,
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
        let runtime = Device::Cpu.runtime().map_err(backend_error)?;
        Ok(Self {
            encoder,
            agent,
            weights,
            tokenizer,
            calibration,
            batch_size: 16,
            device: Device::Cpu,
            runtime,
            cache: TensorCache::new(),
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

    /// [`Self::load`] and run the tenferro forward on `device`
    /// (see [`Self::with_device`]).
    pub fn load_with_device(directory: impl AsRef<Path>, device: Device) -> Result<Self> {
        Self::load(directory)?.with_device(device)
    }

    /// Resolve a Hub snapshot and load its weights, tokenizer, and calibration.
    ///
    /// Network access is opt-in through the `hub` feature and is delegated to
    /// `hf-fetch`. Pass an offline Hub to require an existing cached snapshot;
    /// customize `spec.revision` to pin a checkpoint.
    #[cfg(feature = "hub")]
    pub fn load_from_hub(hub: &hf_fetch::Hub, spec: &hf_fetch::CheckpointSpec) -> Result<Self> {
        let directory = hub
            .resolve(spec)
            .map_err(|error| DecisionError::Transport {
                message: "failed to resolve Laya checkpoint from the Hub".into(),
                source: Some(Box::new(error)),
            })?;
        Self::load(directory)
    }

    /// Set the maximum number of question rows per forward (default: 16).
    pub fn with_batch_size(mut self, batch_size: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(DecisionError::invalid_field(
                "laya.batch_size",
                "must be positive",
            ));
        }
        self.batch_size = batch_size;
        Ok(self)
    }

    /// Run the tenferro forward on `device`, replacing the runtime.
    ///
    /// Weights are uploaded to the new runtime once, on the first forward, and
    /// stay resident in the engine's tensor cache; each later forward uploads
    /// the token batch and masks and downloads the logits. Creating a CUDA
    /// runtime fails (without CPU fallback) when the `cuda` feature is off or
    /// no device is available.
    pub fn with_device(mut self, device: Device) -> Result<Self> {
        self.runtime = device.runtime().map_err(backend_error)?;
        self.device = device;
        self.cache = TensorCache::new();
        Ok(self)
    }

    /// The device the tenferro forward runs on.
    pub fn device(&self) -> Device {
        self.device
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
    pub fn decide(&mut self, state: &State, questions: &QuestionSet) -> Result<Vec<LayaDecision>> {
        state.validate()?;
        questions.validate()?;
        let mut answers = Vec::with_capacity(questions.len());
        for chunk in questions.questions().chunks(self.batch_size) {
            answers.extend(self.decide_batch(state, chunk)?);
        }
        Ok(answers)
    }

    fn decide_batch(
        &mut self,
        state: &State,
        questions: &[(QuestionId, Question)],
    ) -> Result<Vec<LayaDecision>> {
        let rows = questions
            .iter()
            .map(|(_, question)| {
                let (ids, markers) = build_sequence(
                    &self.tokenizer,
                    state,
                    question,
                    self.agent.max_len,
                    self.agent.head_max_len,
                )?;
                let expected = match question {
                    Question::Choice(q) => q.criteria.len(),
                    Question::Score(q) => q.criteria.len(),
                    Question::Noul(_) => 2,
                };
                if markers.len() != expected {
                    return Err(DecisionError::invalid_field(
                        "laya.markers",
                        "the token budget did not preserve every option marker",
                    ));
                }
                Ok((ids, markers, question_type(question)))
            })
            .collect::<Result<Vec<_>>>()?;
        let batch = rows.len();
        let length = rows.iter().map(|row| row.0.len()).max().unwrap();
        let slots = rows.iter().map(|row| row.1.len()).max().unwrap().max(2);
        let mut ids = vec![self.tokenizer.pad_token_id(); length * batch];
        let mut mask = vec![false; length * batch];
        let mut marker_pos = vec![0; slots * batch];
        let mut marker_mask = vec![false; slots * batch];
        let mut qtypes = Vec::with_capacity(batch);
        for (row, (tokens, markers, qtype)) in rows.iter().enumerate() {
            ids[row * length..row * length + tokens.len()].copy_from_slice(tokens);
            mask[row * length..row * length + tokens.len()].fill(true);
            for (slot, position) in markers.iter().enumerate() {
                marker_pos[row * slots + slot] = *position as i64;
                marker_mask[row * slots + slot] = true;
            }
            qtypes.push(qtype.index() as i64);
        }
        let (logits, action) = self
            .runtime
            .with_eager_session(|session| {
                forward_tenferro_cached(
                    session,
                    &mut self.cache,
                    &self.encoder,
                    &self.agent,
                    &self.weights,
                    &ids,
                    &mask,
                    &marker_pos,
                    &marker_mask,
                    &qtypes,
                )
            })
            .map_err(forward_error)?
            .map_err(forward_error)?;
        let action_count = self.agent.action_count();
        if logits.len() != slots * batch || action.len() != action_count * batch {
            return Err(DecisionError::invalid_field(
                "laya.forward",
                "unexpected batched output shape",
            ));
        }
        questions
            .iter()
            .enumerate()
            .map(|(row, (_, question))| {
                let used = rows[row].1.len();
                self.answer(
                    question,
                    rows[row].2,
                    &logits[row * slots..row * slots + used],
                    &action[row * action_count..(row + 1) * action_count],
                )
            })
            .collect()
    }

    fn answer(
        &self,
        question: &Question,
        qtype: QType,
        logits: &[f32],
        action: &[f32],
    ) -> Result<LayaDecision> {
        let used = logits.len();
        let active: Vec<f64> = logits.iter().map(|value| f64::from(*value)).collect();
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

/// Map a tenferro backend-construction error into a decision error.
fn backend_error(error: tenferro_ad::Error) -> DecisionError {
    DecisionError::Backend {
        message: format!("failed to create the tenferro runtime: {error}"),
        source: Some(Box::new(error)),
    }
}

/// Map a tenferro forward error into a decision error.
fn forward_error(error: tenferro_ad::Error) -> DecisionError {
    DecisionError::Backend {
        message: format!("tenferro forward failed: {error}"),
        source: Some(Box::new(error)),
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
