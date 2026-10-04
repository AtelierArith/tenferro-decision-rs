//! Production-Laya parity against `extern/Laya.jl`.
//!
//! Skipped unless both the Julia reference (`fixtures/laya-real/reference.json`,
//! from `tools/gen_laya_real_reference.jl`) and a resolved
//! `convaiinnovations/laya` snapshot in the Hub cache (see `hf-fetch laya`) are
//! present, so a plain checkout stays green.

use std::path::PathBuf;

use hf_fetch::{CheckpointSpec, Hub};
use laya_infer::checkpoint::load_checkpoint;
use laya_infer::model::forward_reference;
use laya_infer::tokenizer::BpeTokenizer;

/// The `convaiinnovations/laya` commit the committed reference was generated
/// from. The fetch preset tracks `main`; this test pins the exact snapshot so
/// it stays reproducible when `main` moves.
const LAYA_REVISION: &str = "7b928d828b7b0e022f929d9bd2e44165aa270148";

fn reference_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/laya-real/reference.json")
}

fn available() -> Option<(PathBuf, serde_json::Value)> {
    let reference = std::fs::read_to_string(reference_path()).ok()?;
    let value: serde_json::Value = serde_json::from_str(&reference).ok()?;
    let mut spec = CheckpointSpec::laya();
    spec.revision = LAYA_REVISION.to_string();
    let mut hub = Hub::from_env();
    hub.offline = true;
    let dir = hub.resolve(&spec).ok()?;
    Some((dir, value))
}

fn column(value: &serde_json::Value) -> Vec<f32> {
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
fn tokenizer_matches_production_reference() {
    let Some((dir, reference)) = available() else {
        eprintln!("skipping: production Laya checkpoint or reference is not present");
        return;
    };
    let tokenizer = BpeTokenizer::from_directory(dir.join("tokenizer")).unwrap();
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
fn forward_matches_production_reference() {
    let Some((dir, reference)) = available() else {
        eprintln!("skipping: production Laya checkpoint or reference is not present");
        return;
    };
    let checkpoint = load_checkpoint(&dir).unwrap();

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

    let expected_logits = column(&reference["logits"]);
    let expected_action = column(&reference["action"]);
    let logit_diff = max_abs_diff(&logits, &expected_logits);
    let action_diff = max_abs_diff(&action, &expected_action);
    // The action head can produce large scores; compare relative to their scale.
    let action_scale = expected_action
        .iter()
        .map(|value| value.abs())
        .fold(1.0f32, f32::max);
    eprintln!("production Laya: logit diff {logit_diff}, action diff {action_diff}");
    assert!(logit_diff < 1e-3, "logits differ by {logit_diff}");
    assert!(
        action_diff < 1e-5 * action_scale,
        "action differs by {action_diff} (scale {action_scale})"
    );
}
