//! End-to-end tests for the prepared-token [`JeffEngine`].
//!
//! The weights are small synthetic tensors, so the suite is hermetic: no
//! checkpoint files and no network.

use decision_core::{
    Answer, ChoiceQuestion, Content, DecisionEngine, NoulCriteria, NoulQuestion, PreparedState,
    Question, QuestionSet, ScoreQuestion, State,
};
use jeff_infer::config::DecisionConfig;
use jeff_infer::engine::JeffEngine;
use jeff_infer::model::{
    forward_reference, AttentionWeights, FullAttentionWeights, JeffConfig, JeffWeights,
    LayerWeights, MlpWeights,
};
use jeff_infer::readout::probabilities;

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }

    fn fill(&mut self, len: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..len).map(|_| lo + (hi - lo) * self.next_f32()).collect()
    }
}

fn config() -> JeffConfig {
    JeffConfig {
        hidden: 8,
        heads: 2,
        head_dim: 4,
        intermediate: 16,
        eps: 1e-5,
    }
}

/// A single full-attention layer is enough for the engine path and keeps the
/// synthetic model small.
fn build_weights(cfg: &JeffConfig, vocab: usize, options: usize, seed: u64) -> JeffWeights {
    let mut rng = Lcg(seed);
    let width = cfg.heads * cfg.head_dim;
    let full = FullAttentionWeights {
        q: rng.fill(cfg.hidden * width, -0.3, 0.3),
        gate: rng.fill(cfg.hidden * width, -0.3, 0.3),
        k: rng.fill(cfg.hidden * width, -0.3, 0.3),
        v: rng.fill(cfg.hidden * width, -0.3, 0.3),
        o: rng.fill(width * cfg.hidden, -0.3, 0.3),
        q_norm: rng.fill(cfg.head_dim, 0.5, 1.5),
        k_norm: rng.fill(cfg.head_dim, 0.5, 1.5),
        rope_theta: 1_000_000.0,
        rotary_dim: 2,
    };
    let layer = LayerWeights {
        input_norm: rng.fill(cfg.hidden, 0.5, 1.5),
        post_norm: rng.fill(cfg.hidden, 0.5, 1.5),
        attention: AttentionWeights::Full(full),
        mlp: MlpWeights {
            gate: rng.fill(cfg.hidden * cfg.intermediate, -0.3, 0.3),
            up: rng.fill(cfg.hidden * cfg.intermediate, -0.3, 0.3),
            down: rng.fill(cfg.intermediate * cfg.hidden, -0.3, 0.3),
        },
    };
    JeffWeights {
        embedding: rng.fill(cfg.hidden * vocab, -1.0, 1.0),
        vocab,
        layers: vec![layer],
        final_norm: rng.fill(cfg.hidden, 0.5, 1.5),
        readout: rng.fill(cfg.hidden * options, -0.3, 0.3),
        options,
    }
}

fn decision(temperature: f64, max_options: usize) -> DecisionConfig {
    DecisionConfig {
        format_version: 1,
        temperature,
        max_options,
    }
}

fn build_engine(options: usize, max_options: usize, temperature: f64, seed: u64) -> JeffEngine {
    JeffEngine::new(
        config(),
        decision(temperature, max_options),
        build_weights(&config(), 12, options, seed),
    )
    .unwrap()
}

/// Three rows with an interior mask hole, one row per question below.
fn prepared() -> PreparedState {
    PreparedState {
        input_ids: vec![vec![3, 1, 4, 2], vec![5, 2, 7, 1], vec![4, 4, 0, 9]],
        attention_mask: vec![
            vec![true, true, true, true],
            vec![true, false, true, true],
            vec![false, true, true, true],
        ],
    }
}

fn choice_question() -> Question {
    Question::Choice(
        ChoiceQuestion::new(
            Content::string("pick one"),
            vec![
                ("a".into(), Content::string("first")),
                ("b".into(), Content::string("second")),
                ("c".into(), Content::string("third")),
            ],
        )
        .unwrap(),
    )
}

fn noul_question() -> Question {
    Question::Noul(
        NoulQuestion::new(
            Content::string("is it so?"),
            NoulCriteria {
                truthy: Some(Content::string("yes")),
                falsy: Some(Content::string("no")),
            },
        )
        .unwrap(),
    )
}

fn score_question() -> Question {
    Question::Score(
        ScoreQuestion::new(
            Content::string("rate it"),
            vec!["low".into(), "mid".into(), "high".into()],
        )
        .unwrap(),
    )
}

fn question_set() -> QuestionSet {
    let mut set = QuestionSet::new();
    set.push("choice", choice_question()).unwrap();
    set.push("noul", noul_question()).unwrap();
    set.push("score", score_question()).unwrap();
    set
}

fn as_f64(logits: &[f32]) -> Vec<f64> {
    logits.iter().map(|value| f64::from(*value)).collect()
}

#[test]
fn answers_choice_noul_and_score_in_order() {
    let temperature = 0.5;
    let mut engine = build_engine(4, 3, temperature, 7);
    let state = State::Prepared(prepared());
    let set = question_set();

    let answers = engine.system_one(&state, &set).unwrap();
    assert_eq!(answers.len(), 3);

    // `logits` and the answer formulas agree with `forward_reference` row by
    // row.
    let prepared = prepared();
    let logits = engine.logits(&prepared).unwrap();
    assert_eq!(logits.len(), 3);
    for (row, (ids, mask)) in prepared
        .input_ids
        .iter()
        .zip(prepared.attention_mask.iter())
        .enumerate()
    {
        let mask_f: Vec<f32> = mask
            .iter()
            .map(|active| if *active { 1.0 } else { 0.0 })
            .collect();
        let start = mask.iter().position(|active| *active).unwrap_or(0);
        let reference = forward_reference(
            engine.config(),
            engine.weights(),
            &ids[start..],
            &mask_f[start..],
        )
        .unwrap();
        assert_eq!(logits[row], reference);
    }

    // Choice: probabilities are the temperature-scaled softmax over columns
    // 0..3 and sum to one.
    match &answers[0] {
        Answer::Choice(answer) => {
            let labels: Vec<String> = answer
                .probabilities
                .iter()
                .map(|(label, _)| label.clone())
                .collect();
            assert_eq!(labels, vec!["a", "b", "c"]);
            let expected = probabilities(&as_f64(&logits[0][..3]), temperature).unwrap();
            let sum: f64 = answer.probabilities.iter().map(|(_, p)| p).sum();
            assert!((sum - 1.0).abs() < 1e-12);

            // The chosen candidate is the argmax.
            let best = expected
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(index, _)| index)
                .unwrap();
            assert_eq!(answer.choice, labels[best]);
            for ((_, got), want) in answer.probabilities.iter().zip(expected.iter()) {
                assert!((got - want).abs() < 1e-12);
            }
        }
        other => panic!("expected a choice answer, found {other:?}"),
    }

    // Noul: the affirmative probability is column 1 of the first two columns.
    match &answers[1] {
        Answer::Noul(answer) => {
            let expected = probabilities(&as_f64(&logits[1][..2]), temperature).unwrap();
            assert!((answer.noul - expected[1]).abs() < 1e-12);
        }
        other => panic!("expected a noul answer, found {other:?}"),
    }

    // Score: legend in criteria order, probabilities over columns 0..3, and
    // the zero-based expected value.
    match &answers[2] {
        Answer::Score(answer) => {
            assert_eq!(answer.legend, vec!["low", "mid", "high"]);
            let expected = probabilities(&as_f64(&logits[2][..3]), temperature).unwrap();
            let sum: f64 = answer.probabilities.iter().sum();
            assert!((sum - 1.0).abs() < 1e-12);
            let score: f64 = answer
                .probabilities
                .iter()
                .enumerate()
                .map(|(index, p)| index as f64 * p)
                .sum();
            assert!((answer.score - score).abs() < 1e-12);
            for (got, want) in answer.probabilities.iter().zip(expected.iter()) {
                assert!((got - want).abs() < 1e-12);
            }
        }
        other => panic!("expected a score answer, found {other:?}"),
    }
}

#[test]
fn temperature_changes_the_distribution() {
    let cold = build_engine(4, 3, 0.25, 7);
    let warm = build_engine(4, 3, 2.0, 7);
    let prepared = prepared();
    let cold_logits = cold.logits(&prepared).unwrap();
    let warm_logits = warm.logits(&prepared).unwrap();
    assert_eq!(cold_logits, warm_logits);

    let cold_p = probabilities(&as_f64(&cold_logits[0][..3]), 0.25).unwrap();
    let warm_p = probabilities(&as_f64(&warm_logits[0][..3]), 2.0).unwrap();
    let cold_max = cold_p.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let warm_max = warm_p.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(cold_max >= warm_max);
}

#[test]
fn rejects_text_and_json_states() {
    let mut engine = build_engine(4, 3, 1.0, 7);
    let set = question_set();
    assert!(engine
        .system_one(&State::Text("natural language".into()), &set)
        .is_err());
    let json = State::Json(Content::object([("prompt", Content::string("hi"))]));
    assert!(engine.system_one(&json, &set).is_err());
}

#[test]
fn rejects_batch_question_count_mismatch() {
    let mut engine = build_engine(4, 3, 1.0, 7);
    let mut set = QuestionSet::new();
    set.push("choice", choice_question()).unwrap();
    set.push("noul", noul_question()).unwrap();
    // Three prepared rows but only two questions.
    assert!(engine
        .system_one(&State::Prepared(prepared()), &set)
        .is_err());
}

#[test]
fn rejects_option_count_above_max_options() {
    // Four candidates but the checkpoint option limit is three.
    let mut engine = build_engine(4, 3, 1.0, 7);
    let mut set = QuestionSet::new();
    set.push(
        "wide",
        Question::Choice(
            ChoiceQuestion::new(
                Content::string("pick one"),
                vec![
                    ("a".into(), Content::string("first")),
                    ("b".into(), Content::string("second")),
                    ("c".into(), Content::string("third")),
                    ("d".into(), Content::string("fourth")),
                ],
            )
            .unwrap(),
        ),
    )
    .unwrap();
    let state = State::Prepared(PreparedState {
        input_ids: vec![vec![1, 2, 3, 4]],
        attention_mask: vec![vec![true; 4]],
    });
    assert!(engine.system_one(&state, &set).is_err());
}

#[test]
fn trims_leading_padding() {
    let engine = build_engine(4, 3, 1.0, 7);
    let padded = PreparedState {
        input_ids: vec![vec![9, 3, 1, 4, 2]],
        attention_mask: vec![vec![false, true, true, true, true]],
    };
    let trimmed = PreparedState {
        input_ids: vec![vec![3, 1, 4, 2]],
        attention_mask: vec![vec![true; 4]],
    };
    assert_eq!(
        engine.logits(&padded).unwrap(),
        engine.logits(&trimmed).unwrap()
    );
}

#[test]
fn constructor_rejects_option_limit_above_readout() {
    let engine = JeffEngine::new(
        config(),
        decision(1.0, 5),
        build_weights(&config(), 12, 4, 7),
    );
    assert!(engine.is_err());
}
