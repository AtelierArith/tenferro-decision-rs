//! Calibration and typed-answer construction.
//!
//! Ports the calibration and answer logic from `extern/Laya.jl/src/agent.jl`
//! and `src/prompt.jl`. Reported values are rounded to four decimal places.

use std::collections::BTreeMap;

use decision_core::{Answer, ChoiceAnswer, DecisionError, NoulAnswer, Result, ScoreAnswer};
use serde_json::Value;

/// Minimum accepted temperature.
pub const TEMP_MIN: f64 = 0.5;
/// Maximum accepted temperature.
pub const TEMP_MAX: f64 = 5.0;

/// One of Laya's three question types, in the checkpoint's numeric order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QType {
    /// `choice` (index 0).
    Choice,
    /// `score` (index 1).
    Score,
    /// `noul` (index 2).
    Noul,
}

impl QType {
    /// Numeric index used by the model's type embedding.
    pub fn index(self) -> usize {
        match self {
            Self::Choice => 0,
            Self::Score => 1,
            Self::Noul => 2,
        }
    }

    /// Checkpoint spelling.
    pub fn name(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Noul => "noul",
        }
    }
}

/// A calibrated decision plus the model's action probability.
#[derive(Clone, Debug, PartialEq)]
pub struct LayaDecision {
    /// The typed answer.
    pub answer: Answer,
    /// Probability of the first action, in `0..=1`.
    pub action_probability: f64,
}

/// Clamp a fitted temperature to `[TEMP_MIN, TEMP_MAX]`, or return `1.0` when
/// it is not finite.
pub fn clamp_temperature(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(TEMP_MIN, TEMP_MAX)
    } else {
        1.0
    }
}

/// Bucket key for `temperature_by_options`, e.g. `choice:6-10`.
pub fn temp_bucket(qtype: QType, options: usize) -> String {
    let size = if options <= 2 {
        "2"
    } else if options <= 5 {
        "3-5"
    } else if options <= 10 {
        "6-10"
    } else {
        "11+"
    };
    format!("{}:{size}", qtype.name())
}

/// Normalized Shannon-entropy confidence `1 - H(p) / log(k)`.
///
/// Returns `1.0` for fewer than two options. Computed in `f32`, matching the
/// reference.
pub fn confidence_from_probs(probabilities: &[f64], k: usize) -> f64 {
    if k < 2 {
        return 1.0;
    }
    let mut entropy = 0.0_f32;
    for probability in probabilities.iter().take(k) {
        let p = (probability.clamp(1e-12, 1.0)) as f32;
        entropy -= p * p.ln();
    }
    let value = 1.0_f32 - entropy / (k as f32).ln();
    value.clamp(0.0, 1.0) as f64
}

/// Round to four decimal places, as upstream reports.
pub fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

/// Numerically stable softmax over a slice.
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|z| (z - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 {
        return vec![0.0; logits.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

/// Softmax probability of the first action.
pub fn action_probability(action_logits: &[f64]) -> f64 {
    if action_logits.is_empty() {
        return 0.0;
    }
    softmax(action_logits)[0]
}

/// Fitted temperatures for the three question types and the per-bucket
/// overrides.
#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
    temperatures: [f64; 3],
    by_options: BTreeMap<String, f64>,
}

impl Default for Calibration {
    fn default() -> Self {
        Self {
            temperatures: [1.0; 3],
            by_options: BTreeMap::new(),
        }
    }
}

impl Calibration {
    /// Build a calibration from raw temperatures, clamping each to
    /// `[TEMP_MIN, TEMP_MAX]`. Rejects non-finite or non-positive inputs.
    pub fn new(temperatures: [f64; 3], by_options: BTreeMap<String, f64>) -> Result<Self> {
        let all_positive = temperatures
            .iter()
            .chain(by_options.values())
            .all(|value| value.is_finite() && *value > 0.0);
        if !all_positive {
            return Err(DecisionError::invalid_field(
                "rl_agent.temperature",
                "calibration temperatures must be finite and positive",
            ));
        }
        Ok(Self {
            temperatures: temperatures.map(clamp_temperature),
            by_options: by_options
                .into_iter()
                .map(|(key, value)| (key, clamp_temperature(value)))
                .collect(),
        })
    }

    /// Parse the calibration fields of `rl_agent_config.json`.
    pub fn from_json_str(text: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(text).map_err(|e| {
            DecisionError::invalid_field("rl_agent", format!("invalid JSON: {e}"))
        })?;
        Self::from_json(&value)
    }

    /// Parse the calibration fields from a value.
    pub fn from_json(value: &Value) -> Result<Self> {
        let object = value.as_object().ok_or_else(|| {
            DecisionError::invalid_field("rl_agent", "config must be a JSON object")
        })?;

        let mut temperatures = [1.0_f64; 3];
        if let Some(items) = object.get("temperature") {
            let items = items.as_array().ok_or_else(|| {
                DecisionError::invalid_field("rl_agent.temperature", "expected an array of three numbers")
            })?;
            if items.len() != 3 {
                return Err(DecisionError::invalid_field(
                    "rl_agent.temperature",
                    "expected exactly three temperatures",
                ));
            }
            for (index, item) in items.iter().enumerate() {
                temperatures[index] = item.as_f64().ok_or_else(|| {
                    DecisionError::invalid_field(
                        format!("rl_agent.temperature[{index}]"),
                        "expected a number",
                    )
                })?;
            }
        }

        let mut by_options = BTreeMap::new();
        if let Some(map) = object.get("temperature_by_options") {
            let map = map.as_object().ok_or_else(|| {
                DecisionError::invalid_field(
                    "rl_agent.temperature_by_options",
                    "expected an object",
                )
            })?;
            for (key, value) in map {
                let value = value.as_f64().ok_or_else(|| {
                    DecisionError::invalid_field(
                        format!("rl_agent.temperature_by_options.{key}"),
                        "expected a number",
                    )
                })?;
                by_options.insert(key.clone(), value);
            }
        }

        Self::new(temperatures, by_options)
    }

    /// Effective temperature for a question type and option count.
    pub fn scale(&self, qtype: QType, options: usize) -> f64 {
        self.by_options
            .get(&temp_bucket(qtype, options))
            .copied()
            .unwrap_or(self.temperatures[qtype.index()])
    }

    /// Calibrated probabilities over the first `k` logits.
    pub fn probabilities(&self, qtype: QType, k: usize, logits: &[f64]) -> Vec<f64> {
        let scale = self.scale(qtype, k);
        let scaled: Vec<f64> = logits.iter().take(k).map(|z| z / scale).collect();
        softmax(&scaled)
    }
}

/// Build a choice answer from calibrated probabilities.
pub fn choice_answer(labels: &[String], probabilities: &[f64]) -> ChoiceAnswer {
    let k = probabilities.len();
    let best = probabilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap_or(0);
    ChoiceAnswer {
        choice: labels.get(best).cloned().unwrap_or_default(),
        probabilities: labels
            .iter()
            .cloned()
            .zip(probabilities.iter().map(|p| round4(*p)))
            .collect(),
        confidence: round4(confidence_from_probs(probabilities, k)),
    }
}

/// Build a score answer from calibrated probabilities.
pub fn score_answer(levels: &[String], probabilities: &[f64]) -> ScoreAnswer {
    let expected: f64 = probabilities
        .iter()
        .enumerate()
        .map(|(index, p)| index as f64 * p)
        .sum();
    ScoreAnswer {
        score: round4(expected),
        legend: levels.to_vec(),
        probabilities: probabilities.iter().map(|p| round4(*p)).collect(),
        confidence: round4(confidence_from_probs(probabilities, probabilities.len())),
    }
}

/// Build a `noul` answer; `probabilities[1]` is the affirmative probability.
pub fn noul_answer(probabilities: &[f64]) -> NoulAnswer {
    let truthy = probabilities.get(1).copied().unwrap_or(0.0);
    NoulAnswer {
        noul: round4(truthy),
    }
}
