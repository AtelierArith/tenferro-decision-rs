//! Jeff [`DecisionEngine`] for prepared tokens and local Text/Json prompts.
//!
//! It accepts [`State::Prepared`] tokens, and Text/Json states when tokenizer
//! assets are loaded. It runs a forward once per row through a
//! workspace reused across rows, and turns the readout logits into typed answers
//! with [`crate::readout`].
//!
//! [`JeffBackend`] selects the forward: the default `Auto` picks the optimized
//! CPU path with prepared tenferro projection extensions; `Host` runs the all-host oracle
//! ([`forward_reference_with`]); `Tenferro` is the backend-portable path (slower
//! on CPU today).
//!
//! Row `i` of the prepared state answers question `i`, so the batch and the
//! [`QuestionSet`] must have the same length. Each question selects its own
//! number of readout columns:
//!
//! - **choice**: one column per candidate id, in criteria order
//! - **noul**: the first two columns (false, true)
//! - **score**: one column per level, in criteria order
//!
//! `load` reads local tokenizer assets and compiles the checkpoint chat template.
//! Constructors from raw weights remain prepared-only until `with_tokenizer`.

use std::sync::Arc;

use decision_core::{
    Answer, DecisionEngine, DecisionError, PreparedState, Question, QuestionSet, Result, State,
};
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;
use tenferro_gated_delta::GatedDeltaWorkspace;
use tenferro_infer::TensorCache;

use crate::config::DecisionConfig;
use crate::host_opt::HostOptWorkspace;
use crate::model::{
    DeltaKernel, JeffConfig, JeffWeights, forward_reference_with, forward_tenferro_cached_kernel,
};
use crate::readout::{choice_answer, noul_answer, score_answer};

/// Which forward the engine runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum JeffBackend {
    /// The production CPU forward ([`HostOpt`]), with a SIMD recurrent scan
    /// and prepared tenferro projection extensions for short inputs.
    ///
    /// [`HostOpt`]: JeffBackend::HostOpt
    #[default]
    Auto,
    /// The all-host fused forward — the correctness oracle.
    Host,
    /// The host-optimized forward: same math as [`JeffBackend::Host`] but rayon
    /// parallel (tokens/heads/elements) with a reusable workspace, ported from
    /// the Julia native CPU runtime, with prepared tenferro projections for
    /// short inputs and the portable provider for longer sequences.
    HostOpt,
    /// The tenferro-native forward — backend-portable, with the DeltaNet
    /// running through the fused host recurrent kernel (`GatedDelta` extension
    /// op) on CPU by default.
    Tenferro,
}

/// A prepared-token Jeff engine: dimensions, decision settings, and weights.
#[derive(Clone, Debug)]
pub struct JeffEngine {
    config: JeffConfig,
    decision: DecisionConfig,
    weights: Arc<JeffWeights>,
    prepared_cpu: crate::prepared_cpu::PreparedCpuModel,
    backend: JeffBackend,
    /// The tenferro runtime the tenferro forward runs on (CPU today).
    runtime: Arc<EagerRuntime>,
    /// DeltaNet scratch reused across rows.
    workspace: GatedDeltaWorkspace,
    /// Activation buffers for the host-optimized forward.
    host_opt: HostOptWorkspace,
    /// Weight tensors cached across rows (tenferro backend).
    cache: TensorCache,
    /// How the tenferro forward runs each Gated DeltaNet layer.
    delta_kernel: DeltaKernel,
    tokenizer: Option<crate::tokenizer::JeffTokenizer>,
}

impl JeffEngine {
    /// Build an engine with the default forward ([`JeffBackend::Auto`]).
    pub fn new(config: JeffConfig, decision: DecisionConfig, weights: JeffWeights) -> Result<Self> {
        Self::with_backend(config, decision, weights, JeffBackend::default())
    }

    /// Build an engine selecting the forward.
    ///
    /// Also requires `decision.max_options` to fit inside the readout, since
    /// every question slices at most that many columns.
    pub fn with_backend(
        config: JeffConfig,
        decision: DecisionConfig,
        weights: JeffWeights,
        backend: JeffBackend,
    ) -> Result<Self> {
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
        let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).map_err(backend_error)?;
        let weights = Arc::new(weights);
        let prepared_cpu =
            crate::prepared_cpu::PreparedCpuModel::from_shared(config, weights.clone())?;
        Ok(Self {
            prepared_cpu,
            config,
            decision,
            weights,
            backend,
            runtime,
            workspace: GatedDeltaWorkspace::new(),
            host_opt: HostOptWorkspace::new(),
            cache: TensorCache::new(),
            delta_kernel: DeltaKernel::default(),
            tokenizer: None,
        })
    }

    /// Load a local checkpoint with the default forward backend.
    pub fn load(directory: impl AsRef<std::path::Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let checkpoint = crate::checkpoint::load_checkpoint(directory)?;
        let engine = Self::new(checkpoint.config, checkpoint.decision, checkpoint.weights)?;
        if directory.join("tokenizer.json").is_file() {
            engine.with_tokenizer(crate::tokenizer::JeffTokenizer::from_directory(directory)?)
        } else {
            Ok(engine)
        }
    }

    /// Resolve a Hub snapshot and load it with the default forward backend.
    ///
    /// Network access is opt-in through the `hub` feature and is delegated to
    /// `hf-fetch`. Pass an offline Hub to require an existing cached snapshot;
    /// customize `spec.revision` to pin a checkpoint.
    #[cfg(feature = "hub")]
    pub fn load_from_hub(hub: &hf_fetch::Hub, spec: &hf_fetch::CheckpointSpec) -> Result<Self> {
        let directory = hub
            .resolve(spec)
            .map_err(|error| DecisionError::Transport {
                message: "failed to resolve Jeff checkpoint from the Hub".into(),
                source: Some(Box::new(error)),
            })?;
        Self::load(directory)
    }

    /// Enable Text/Json states using validated offline tokenizer assets.
    pub fn with_tokenizer(mut self, tokenizer: crate::tokenizer::JeffTokenizer) -> Result<Self> {
        if tokenizer.option_limit() < self.decision.max_options {
            return Err(DecisionError::invalid_field(
                "jeff.tokenizer",
                "insufficient checkpoint answer codes",
            ));
        }
        self.tokenizer = Some(tokenizer);
        Ok(self)
    }

    /// Render and encode Text/Json states into one row per question.
    pub fn prepare(&self, state: &State, questions: &QuestionSet) -> Result<PreparedState> {
        state.validate()?;
        questions.validate()?;
        if let State::Prepared(prepared) = state {
            if prepared.input_ids.len() != questions.len() {
                return Err(DecisionError::invalid_field(
                    "questions",
                    "expected one question per prepared row",
                ));
            }
            return Ok(prepared.clone());
        }
        self.tokenizer
            .as_ref()
            .ok_or_else(|| {
                DecisionError::unsupported(
                    "Text/Json states require Jeff tokenizer assets; use load or with_tokenizer",
                )
            })?
            .prepare(state, questions)
    }

    /// Select how the tenferro forward runs each Gated DeltaNet layer.
    pub fn with_delta_kernel(mut self, kernel: DeltaKernel) -> Self {
        self.delta_kernel = kernel;
        self
    }

    /// The selected forward.
    pub fn backend(&self) -> JeffBackend {
        self.backend
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
    pub fn logits(&mut self, state: &PreparedState) -> Result<Vec<Vec<f32>>> {
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
    fn row_logits(&mut self, ids: &[i64], mask: &[bool]) -> Result<Vec<f32>> {
        let start = mask.iter().position(|active| *active).unwrap_or(0);
        let ids = &ids[start..];
        let mask: Vec<f32> = mask[start..]
            .iter()
            .map(|active| if *active { 1.0 } else { 0.0 })
            .collect();
        match self.backend {
            JeffBackend::Auto | JeffBackend::HostOpt => {
                self.prepared_cpu.forward(&mut self.host_opt, ids, &mask)
            }
            JeffBackend::Host => {
                forward_reference_with(&mut self.workspace, &self.config, &self.weights, ids, &mask)
            }
            JeffBackend::Tenferro => self
                .runtime
                .with_eager_session(|session| {
                    forward_tenferro_cached_kernel(
                        &mut self.workspace,
                        &mut self.cache,
                        session,
                        &self.config,
                        &self.weights,
                        ids,
                        &mask,
                        self.delta_kernel,
                    )
                })
                .map_err(forward_error)?
                .map_err(forward_error),
        }
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

/// Map a tenferro backend-construction error into a decision error.
fn backend_error(error: tenferro_ad::Error) -> DecisionError {
    DecisionError::Backend {
        message: format!("failed to create the tenferro CPU runtime: {error}"),
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

impl DecisionEngine for JeffEngine {
    type Error = DecisionError;

    fn system_one(&mut self, state: &State, questions: &QuestionSet) -> Result<Vec<Answer>> {
        let encoded;
        let prepared = match state {
            State::Prepared(prepared) => prepared,
            State::Text(_) | State::Json(_) => {
                encoded = self.prepare(state, questions)?;
                &encoded
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
