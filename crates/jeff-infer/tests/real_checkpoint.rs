//! Production-Jeff parity against `extern/JeffClient.jl`.
//!
//! Ignored by default: it loads the ~1.7 GB `mstrasser/Jeff-Qwen3.5-0.8B`
//! checkpoint and runs a full 0.8B forward, so it is meant to be run manually:
//!
//! ```text
//! cargo test --release -p jeff-infer --test real_checkpoint -- --ignored --nocapture
//! ```
//!
//! It also requires the Julia reference (`fixtures/jeff-real/reference.json`,
//! from `tools/gen_jeff_real_reference.jl`) and a resolved snapshot in the Hub
//! cache (`hf-fetch jeff`); when either is absent it skips.

use std::path::PathBuf;

use hf_fetch::{CheckpointSpec, Hub};
use jeff_infer::checkpoint::load_checkpoint;
use jeff_infer::host_opt::forward_host_opt;
use jeff_infer::model::forward_reference;

/// The `mstrasser/Jeff-Qwen3.5-0.8B` commit the committed reference was
/// generated from (the `JeffClient.jl` pin).
const JEFF_REVISION: &str = "0f212b3e72acb4dde3f7da61e925d6ab7f819990";

fn reference_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jeff-real/reference.json")
}

fn available() -> Option<(PathBuf, serde_json::Value)> {
    let reference = std::fs::read_to_string(reference_path()).ok()?;
    let value: serde_json::Value = serde_json::from_str(&reference).ok()?;
    let mut spec = CheckpointSpec::jeff();
    spec.revision = JEFF_REVISION.to_string();
    let mut hub = Hub::from_env();
    hub.offline = true;
    let dir = hub.resolve(&spec).ok()?;
    Some((dir, value))
}

fn f32_array(value: &serde_json::Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

#[test]
#[ignore = "loads the ~1.7 GB production Jeff checkpoint; run with --release -- --ignored"]
fn forward_matches_production_reference() {
    let Some((dir, reference)) = available() else {
        eprintln!("skipping: production Jeff checkpoint or reference is not present");
        return;
    };
    let checkpoint = load_checkpoint(&dir).unwrap();
    let (cfg, weights) = (checkpoint.config, checkpoint.weights);

    let ids: Vec<i64> = reference["input_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let mask = f32_array(&reference["attention_mask"]);
    let expected = f32_array(&reference["logits"]);

    let got = forward_reference(&cfg, &weights, &ids, &mask).unwrap();
    assert_eq!(got.len(), expected.len(), "readout column count");

    let diff = got
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let scale = expected.iter().map(|v| v.abs()).fold(1.0f32, f32::max);
    eprintln!("production Jeff: logit diff {diff} (scale {scale})");
    assert!(
        diff < 1e-3 * scale,
        "logits differ by {diff} (scale {scale})"
    );

    // The host-optimized path must match the oracle (and thus the reference).
    let optimized = forward_host_opt(&cfg, &weights, &ids, &mask).unwrap();
    let opt_diff = optimized
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("production Jeff host-opt vs oracle: logit diff {opt_diff} (scale {scale})");
    assert!(
        opt_diff < 1e-4 * scale,
        "host-opt logits differ from the oracle by {opt_diff} (scale {scale})"
    );
}
