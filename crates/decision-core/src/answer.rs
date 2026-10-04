use crate::{DecisionError, Result};

/// The answer to a [`ChoiceQuestion`](crate::ChoiceQuestion).
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceAnswer {
    /// The selected candidate id.
    pub choice: String,
    /// Candidate id → probability, in criteria order.
    pub probabilities: Vec<(String, f64)>,
    /// Confidence in `0..=1`, as defined by the answering engine.
    pub confidence: f64,
}

/// The answer to a [`NoulQuestion`](crate::NoulQuestion).
#[derive(Clone, Debug, PartialEq)]
pub struct NoulAnswer {
    /// Probability of the affirmative outcome, in `0..=1`.
    pub noul: f64,
}

/// The answer to a [`ScoreQuestion`](crate::ScoreQuestion).
#[derive(Clone, Debug, PartialEq)]
pub struct ScoreAnswer {
    /// Zero-based expected score.
    pub score: f64,
    /// Level labels in order.
    pub legend: Vec<String>,
    /// Level probabilities in order.
    pub probabilities: Vec<f64>,
    /// Confidence in `0..=1`.
    pub confidence: f64,
}

impl ScoreAnswer {
    /// Validate the answer's internal consistency.
    pub fn validate(&self) -> Result<()> {
        if self.legend.len() != self.probabilities.len() {
            return Err(DecisionError::invalid_field(
                "score",
                "legend and probabilities must have the same length",
            ));
        }
        for (index, probability) in self.probabilities.iter().enumerate() {
            if !(0.0..=1.0).contains(probability) {
                return Err(DecisionError::invalid_field(
                    format!("score.probabilities[{index}]"),
                    "probability must be in 0..=1",
                ));
            }
        }
        Ok(())
    }
}

/// A typed decision answer.
#[derive(Clone, Debug, PartialEq)]
pub enum Answer {
    /// Answer to a choice question.
    Choice(ChoiceAnswer),
    /// Answer to a `noul` question.
    Noul(NoulAnswer),
    /// Answer to a score question.
    Score(ScoreAnswer),
}
