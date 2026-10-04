//! Real-checkpoint parity against `extern/Laya.jl`.
//!
//! `fixtures/laya-tiny/` is written by `tools/gen_laya_fixture.jl` (seeded
//! `Laya.write_tiny_checkpoint`) and `reference.json` records the Julia
//! `DecisionModel` outputs for a fixed prepared batch plus tokenizer goldens.
//!
//! If the fixture is absent the tests skip, so a plain checkout stays green.

use std::path::{Path, PathBuf};

use laya_infer::checkpoint::load_checkpoint;
use laya_infer::model::forward_reference;
use laya_infer::tokenizer::BpeTokenizer;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/laya-tiny")
}

fn available() -> Option<PathBuf> {
    let dir = fixture_dir();
    if dir.join("model.safetensors").is_file() && dir.join("reference.json").is_file() {
        Some(dir)
    } else {
        None
    }
}

fn reference(dir: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap()).unwrap()
}

fn f32_column(value: &serde_json::Value) -> Vec<f32> {
    value[0]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn tokenizer_matches_julia_reference() {
    let Some(dir) = available() else {
        eprintln!("skipping: fixtures/laya-tiny is not present");
        return;
    };
    let tokenizer = BpeTokenizer::from_directory(dir.join("tokenizer")).unwrap();
    let reference = reference(&dir);
    let samples = reference["tokenizer_samples"].as_array().unwrap();
    let goldens = reference["tokenizer_ids"].as_array().unwrap();
    for (sample, golden) in samples.iter().zip(goldens) {
        let text = sample.as_str().unwrap();
        let expected: Vec<i64> = golden
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        assert_eq!(
            tokenizer.encode(text),
            expected,
            "tokenizer mismatch for {text:?}"
        );
    }
}

#[test]
fn forward_matches_julia_reference() {
    let Some(dir) = available() else {
        eprintln!("skipping: fixtures/laya-tiny is not present");
        return;
    };
    let checkpoint = load_checkpoint(&dir).unwrap();
    let reference = reference(&dir);

    let ids: Vec<i64> = reference["input_ids"][0]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let mask: Vec<bool> = reference["attention_mask"][0]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect();
    let marker_pos: Vec<i64> = reference["marker_pos"][0]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let marker_mask: Vec<bool> = reference["marker_mask"][0]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect();
    let qtype: Vec<i64> = reference["qtype"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();

    let (logits, action) = forward_reference(
        &checkpoint.encoder,
        &checkpoint.agent,
        &checkpoint.weights,
        &ids,
        &mask,
        &marker_pos,
        &marker_mask,
        &qtype,
    )
    .unwrap();

    let expected_logits = f32_column(&reference["logits"]);
    let expected_action = f32_column(&reference["action"]);
    let logit_diff = max_abs_diff(&logits, &expected_logits);
    let action_diff = max_abs_diff(&action, &expected_action);
    eprintln!("laya fixture: logit diff {logit_diff}, action diff {action_diff}");
    assert!(
        logit_diff < 1e-5,
        "logits differ: {logits:?} vs {expected_logits:?}"
    );
    assert!(
        action_diff < 1e-5,
        "action differs: {action:?} vs {expected_action:?}"
    );
}

#[test]
fn engine_loads_a_real_julia_checkpoint() {
    let Some(dir) = available() else {
        eprintln!("skipping: fixtures/laya-tiny is not present");
        return;
    };
    use decision_core::{ChoiceQuestion, Content, DecisionEngine, Question, QuestionSet, State};
    use laya_infer::agent::LayaEngine;

    let mut engine = LayaEngine::load(&dir).unwrap();

    let mut set = QuestionSet::new();
    set.push(
        "q",
        Question::Choice(
            ChoiceQuestion::new(
                Content::string("pick one"),
                vec![
                    ("a".into(), Content::string("first")),
                    ("b".into(), Content::string("second")),
                ],
            )
            .unwrap(),
        ),
    )
    .unwrap();
    let answers = engine
        .system_one(&State::Text("hello world".to_string()), &set)
        .unwrap();
    assert_eq!(answers.len(), 1);
}
