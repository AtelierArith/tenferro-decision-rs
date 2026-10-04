//! Real-checkpoint parity: the synthetic Qwen3.5 fixture committed under the
//! `extern/JeffClient.jl` submodule (lengths 1/3/63/64/65, batch 2, left
//! padding) against the independent PyTorch reference logits.
//!
//! The fixture lives in a submodule. When the submodule is not initialized the
//! test skips instead of failing, so a plain checkout still runs green; CI that
//! runs `git submodule update --init` exercises the real parity.

use std::path::{Path, PathBuf};

use jeff_infer::checkpoint::load_checkpoint;
use jeff_infer::model::{forward_reference, forward_tenferro};
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../extern/JeffClient.jl/test/fixtures/native")
}

fn available() -> Option<PathBuf> {
    let dir = fixture_dir();
    if dir.join("model.safetensors").is_file() && dir.join("reference.json").is_file() {
        Some(dir)
    } else {
        None
    }
}

/// Trim leading zero-mask positions, mirroring the reference's left-padding
/// policy (positions are rebased to zero over the active span).
fn active_span(ids: &[i64], mask: &[f32]) -> (Vec<i64>, Vec<f32>) {
    let start = mask.iter().position(|m| *m != 0.0).unwrap_or(0);
    (ids[start..].to_vec(), vec![1.0f32; ids.len() - start])
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

struct Case {
    ids: Vec<i64>,
    mask: Vec<f32>,
    expected: Vec<f32>,
}

fn load_cases(dir: &Path) -> Vec<Case> {
    let reference: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap())
            .unwrap();
    let mut cases = Vec::new();
    for case in reference.as_array().unwrap() {
        let inputs = &case["inputs"];
        let ids = inputs["input_ids"].as_array().unwrap();
        let mask = inputs["attention_mask"].as_array().unwrap();
        let logits = case["logits"].as_array().unwrap();
        for row in 0..ids.len() {
            cases.push(Case {
                ids: ids[row]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_i64().unwrap())
                    .collect(),
                mask: mask[row]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect(),
                expected: logits[row]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect(),
            });
        }
    }
    cases
}

#[test]
fn host_reference_matches_pytorch_fixture() {
    let Some(dir) = available() else {
        eprintln!("skipping: JeffClient native fixture submodule is not initialized");
        return;
    };
    let checkpoint = load_checkpoint(&dir).expect("load checkpoint");
    let (cfg, weights) = (checkpoint.config, checkpoint.weights);
    for case in load_cases(&dir) {
        let (ids, mask) = active_span(&case.ids, &case.mask);
        let got = forward_reference(&cfg, &weights, &ids, &mask).unwrap();
        let diff = max_abs_diff(&got, &case.expected);
        assert!(
            diff < 1e-4,
            "host forward differs by {diff} at length {}; got {got:?}, want {:?}",
            ids.len(),
            case.expected
        );
    }
}

#[test]
fn tenferro_forward_matches_pytorch_fixture() {
    let Some(dir) = available() else {
        eprintln!("skipping: JeffClient native fixture submodule is not initialized");
        return;
    };
    let checkpoint = load_checkpoint(&dir).expect("load checkpoint");
    let (cfg, weights) = (checkpoint.config, checkpoint.weights);
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    for case in load_cases(&dir) {
        let (ids, mask) = active_span(&case.ids, &case.mask);
        let got = runtime
            .with_eager_session(|session| forward_tenferro(session, &cfg, &weights, &ids, &mask))
            .unwrap()
            .unwrap();
        let diff = max_abs_diff(&got, &case.expected);
        assert!(
            diff < 1e-3,
            "tenferro forward differs by {diff} at length {}; got {got:?}, want {:?}",
            ids.len(),
            case.expected
        );
    }
}
