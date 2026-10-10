#![cfg(feature = "cuda")]

//! CUDA hardware gates for the Laya tenferro forward (`--features cuda`).
//!
//! Ignored by default: they need a CUDA device, cuBLAS and cuTENSOR, plus the
//! pinned `convaiinnovations/laya` snapshot (`hf-fetch laya`); without the
//! snapshot they skip. A missing device is a failure. Run with
//!
//! ```text
//! CUDA_VISIBLE_DEVICES=1 cargo test --release -p laya-infer --features cuda \
//!     --test cuda -- --ignored --test-threads=1
//! ```

use std::path::PathBuf;

use hf_fetch::{CheckpointSpec, Hub};
use laya_infer::agent::{Device, LayaEngine};
use laya_infer::model::{TensorCache, forward_tenferro_cached};
use serde_json::Value;

const LAYA_REVISION: &str = "7b928d828b7b0e022f929d9bd2e44165aa270148";

fn checkpoint_dir() -> Option<PathBuf> {
    let mut spec = CheckpointSpec::laya();
    spec.revision = LAYA_REVISION.to_string();
    let mut hub = Hub::from_env();
    hub.offline = true;
    hub.resolve(&spec).ok()
}

fn fixture(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/laya-real")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn integers(value: &Value) -> Vec<i64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect()
}

fn booleans(value: &Value) -> Vec<bool> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect()
}

fn floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            assert!(x.is_finite() && y.is_finite());
            (x - y).abs()
        })
        .fold(0.0, f32::max)
}

type Batch = (Vec<i64>, Vec<bool>, Vec<i64>, Vec<bool>, Vec<i64>);

#[test]
#[ignore = "requires CUDA hardware and the production Laya checkpoint; run with --release -- --ignored"]
fn cuda_production_logits_match_cpu_and_julia() {
    let Some(dir) = checkpoint_dir() else {
        eprintln!("skipping: production Laya checkpoint is not present");
        return;
    };
    let engine = LayaEngine::load(&dir).unwrap();
    let cpu = Device::Cpu.runtime().unwrap();
    let gpu = Device::Cuda(0).runtime().expect("CUDA device required");

    // The single-row Julia forward reference and the batched (B5, padded)
    // bundled-email capture.
    let single = fixture("reference.json");
    let answers = fixture("answers.json");
    let batch = &answers["batch"];
    let cases: Vec<(&str, Batch, Vec<f32>, Vec<f32>)> = vec![
        (
            "reference-B1",
            (
                integers(&single["input_ids"][0]),
                booleans(&single["attention_mask"][0]),
                integers(&single["marker_pos"][0]),
                booleans(&single["marker_mask"][0]),
                integers(&single["qtype"]),
            ),
            floats(&single["logits"][0]),
            floats(&single["action"][0]),
        ),
        (
            "answers-batch",
            (
                integers(&batch["input_ids"]),
                booleans(&batch["attention_mask"]),
                integers(&batch["marker_pos"]),
                booleans(&batch["marker_mask"]),
                integers(&batch["qtype"]),
            ),
            floats(&answers["logits"]),
            floats(&answers["action"]),
        ),
    ];
    let mut cpu_cache = TensorCache::new();
    let mut gpu_cache = TensorCache::new();
    for (name, (ids, mask, marker_pos, marker_mask, qtype), julia_logits, julia_action) in cases {
        let run = |runtime: &std::sync::Arc<tenferro_ad::EagerRuntime>, cache: &mut TensorCache| {
            runtime
                .with_eager_session(|session| {
                    forward_tenferro_cached(
                        session,
                        cache,
                        engine.encoder(),
                        engine.agent(),
                        engine.weights(),
                        &ids,
                        &mask,
                        &marker_pos,
                        &marker_mask,
                        &qtype,
                    )
                })
                .unwrap()
                .unwrap()
        };
        let (cpu_logits, cpu_action) = run(&cpu, &mut cpu_cache);
        // Repeat on the device to exercise the resident weight cache.
        for request in 0..2 {
            let (logits, action) = run(&gpu, &mut gpu_cache);
            let scale = julia_action.iter().map(|v| v.abs()).fold(1.0f32, f32::max);
            let logit_cpu = max_diff(&logits, &cpu_logits);
            let action_cpu = max_diff(&action, &cpu_action);
            let logit_julia = max_diff(&logits, &julia_logits);
            let action_julia = max_diff(&action, &julia_action);
            eprintln!(
                "{name} request {request}: logits cuda-vs-cpu {logit_cpu:e} cuda-vs-julia \
                 {logit_julia:e}; action cuda-vs-cpu {action_cpu:e} cuda-vs-julia \
                 {action_julia:e} (action scale {scale})"
            );
            assert!(
                logit_cpu <= 1e-4,
                "{name}: logits differ from CPU by {logit_cpu}"
            );
            assert!(
                action_cpu <= 1e-4 * scale,
                "{name}: action differs from CPU by {action_cpu}"
            );
            assert!(logit_julia < 2e-3, "{name}: logits differ from Julia");
        }
    }
}

#[test]
#[ignore = "requires CUDA hardware and the production Laya checkpoint; run with --release -- --ignored"]
fn cuda_engine_decisions_match_cpu_engine() {
    use decision_core::{Content, NoulCriteria, NoulQuestion, Question, QuestionSet, State};
    let Some(dir) = checkpoint_dir() else {
        eprintln!("skipping: production Laya checkpoint is not present");
        return;
    };
    let mut cpu = LayaEngine::load(&dir).unwrap();
    let mut gpu =
        LayaEngine::load_with_device(&dir, Device::Cuda(0)).expect("CUDA device required");
    assert_eq!(gpu.device(), Device::Cuda(0));
    let state = State::Text(
        "Customer reports the parcel arrived crushed and asks for a replacement.".into(),
    );
    let mut questions = QuestionSet::new();
    for (id, text) in [
        ("damaged", "Is the parcel damaged?"),
        ("refund", "Does the customer ask for a refund?"),
        ("urgent", "Is this urgent?"),
    ] {
        questions
            .push(
                id,
                Question::Noul(
                    NoulQuestion::new(
                        Content::string(text),
                        NoulCriteria {
                            truthy: Some(Content::Null),
                            falsy: None,
                        },
                    )
                    .unwrap(),
                ),
            )
            .unwrap();
    }
    let expected = cpu.decide(&state, &questions).unwrap();
    for _ in 0..2 {
        let got = gpu.decide(&state, &questions).unwrap();
        for (a, b) in got.iter().zip(&expected) {
            assert!((a.action_probability - b.action_probability).abs() < 1e-5);
            match (&a.answer, &b.answer) {
                (decision_core::Answer::Noul(a), decision_core::Answer::Noul(b)) => {
                    assert!((a.noul - b.noul).abs() < 1e-5, "{} vs {}", a.noul, b.noul);
                }
                other => panic!("unexpected answers {other:?}"),
            }
        }
    }
}
