//! Jeff's trained readout: temperature-scaled softmax and the answer formulas.
//!
//! Ports `extern/JeffClient.jl/src/questions.jl`.

use decision_core::{ChoiceAnswer, DecisionError, NoulAnswer, Result, ScoreAnswer};

/// Temperature-scaled stable softmax over the active option logits.
pub fn probabilities(logits: &[f64], temperature: f64) -> Result<Vec<f64>> {
    if !(temperature.is_finite() && temperature > 0.0) {
        return Err(DecisionError::invalid_field(
            "temperature",
            "must be finite and positive",
        ));
    }
    if logits.is_empty() {
        return Err(DecisionError::invalid_field(
            "logits",
            "at least one option is required",
        ));
    }
    if logits.iter().any(|z| !z.is_finite()) {
        return Err(DecisionError::invalid_field(
            "logits",
            "active option logits must be finite",
        ));
    }
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logits
        .iter()
        .map(|z| ((z - max) / temperature).exp())
        .collect();
    let sum: f64 = weights.iter().sum();
    Ok(weights.into_iter().map(|w| w / sum).collect())
}

/// Choice answer with Jeff's confidence:
/// `clamp((p_best - 1/n) / (1 - 1/n), 0, 1)`.
pub fn choice_answer(
    labels: &[String],
    logits: &[f64],
    temperature: f64,
) -> Result<ChoiceAnswer> {
    let probabilities = probabilities(logits, temperature)?;
    let best = argmax(&probabilities);
    let n = probabilities.len();
    let confidence = if n == 1 {
        1.0
    } else {
        let n = n as f64;
        ((probabilities[best] - 1.0 / n) / (1.0 - 1.0 / n)).clamp(0.0, 1.0)
    };
    Ok(ChoiceAnswer {
        choice: labels.get(best).cloned().unwrap_or_default(),
        probabilities: labels
            .iter()
            .cloned()
            .zip(probabilities.iter().copied())
            .collect(),
        confidence,
    })
}

/// `noul` answer: the affirmative probability (model columns are false, true).
pub fn noul_answer(logits: &[f64], temperature: f64) -> Result<NoulAnswer> {
    let probabilities = probabilities(logits, temperature)?;
    if probabilities.len() < 2 {
        return Err(DecisionError::invalid_field(
            "logits",
            "a noul question needs two columns (false, true)",
        ));
    }
    Ok(NoulAnswer {
        noul: probabilities[1],
    })
}

/// Score answer with Jeff's confidence:
/// `max(0, 1 - distance / baseline)`, where `distance` is the expected
/// absolute distance from the most likely level and `baseline` is the mean
/// distance from the midpoint.
pub fn score_answer(
    levels: &[String],
    logits: &[f64],
    temperature: f64,
) -> Result<ScoreAnswer> {
    let probabilities = probabilities(logits, temperature)?;
    let n = probabilities.len();
    let best = argmax(&probabilities) as f64;
    let distance: f64 = probabilities
        .iter()
        .enumerate()
        .map(|(index, p)| p * (index as f64 - best).abs())
        .sum();
    let midpoint = (n as f64 - 1.0) / 2.0;
    let baseline: f64 = (0..n)
        .map(|index| (index as f64 - midpoint).abs())
        .sum::<f64>()
        / n as f64;
    let confidence = if baseline > 0.0 {
        (1.0 - distance / baseline).max(0.0)
    } else {
        1.0
    };
    let score: f64 = probabilities
        .iter()
        .enumerate()
        .map(|(index, p)| index as f64 * p)
        .sum();
    Ok(ScoreAnswer {
        score,
        legend: levels.to_vec(),
        probabilities,
        confidence,
    })
}

fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap_or(0)
}
