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
use jeff_infer::model::{DeltaKernel, forward_reference, forward_tenferro_cached_kernel};
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;
use tenferro_gated_delta::GatedDeltaWorkspace;
use tenferro_infer::TensorCache;

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

    // Exercise the native constants cache across separate requests on the
    // production weights, comparing both executions with the Julia logits.
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let mut workspace = GatedDeltaWorkspace::new();
    let mut cache = TensorCache::new();
    for request in 0..2 {
        let native = runtime
            .with_eager_session(|session| {
                forward_tenferro_cached_kernel(
                    &mut workspace,
                    &mut cache,
                    session,
                    &cfg,
                    &weights,
                    &ids,
                    &mask,
                    DeltaKernel::TensorNative,
                )
            })
            .unwrap()
            .unwrap();
        assert_eq!(native.len(), expected.len());
        let diff = native
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "production Jeff tensor-native request {request}: logit diff {diff} (scale {scale})"
        );
        assert!(diff < 1e-3 * scale, "native logits differ by {diff}");
    }
}

#[test]
#[ignore = "loads the ~1.7 GB production Jeff checkpoint; run with --release -- --ignored"]
fn engine_answers_match_production_julia_decide() {
    use decision_core::{
        Answer, ChoiceQuestion, Content, DecisionEngine, NoulCriteria, NoulQuestion, PreparedState,
        Question, QuestionSet, ScoreQuestion, State,
    };
    use jeff_infer::engine::JeffEngine;
    let Some((dir, reference)) = available() else {
        eprintln!("skipping: production Jeff checkpoint or reference is not present");
        return;
    };
    let cases = reference["answer_cases"]
        .as_array()
        .expect("regenerate the Julia reference to include decide answers");
    assert_eq!(cases.len(), 3);
    let checkpoint = load_checkpoint(&dir).unwrap();
    assert_eq!(
        checkpoint.decision.temperature,
        reference["temperature"].as_f64().unwrap()
    );
    let mut engine =
        JeffEngine::new(checkpoint.config, checkpoint.decision, checkpoint.weights).unwrap();
    let mut questions = QuestionSet::new();
    questions
        .push(
            "choice",
            Question::Choice(
                ChoiceQuestion::new(
                    Content::string("prepared-token question"),
                    vec![
                        ("a".into(), Content::string("first")),
                        ("b".into(), Content::string("second")),
                        ("c".into(), Content::string("third")),
                    ],
                )
                .unwrap(),
            ),
        )
        .unwrap();
    questions
        .push(
            "noul",
            Question::Noul(
                NoulQuestion::new(
                    Content::string("prepared-token question"),
                    NoulCriteria {
                        truthy: Some(Content::string("true")),
                        falsy: Some(Content::string("false")),
                    },
                )
                .unwrap(),
            ),
        )
        .unwrap();
    questions
        .push(
            "score",
            Question::Score(
                ScoreQuestion::new(
                    Content::string("prepared-token question"),
                    vec!["low".into(), "middle".into(), "high".into()],
                )
                .unwrap(),
            ),
        )
        .unwrap();
    let prepared = PreparedState {
        input_ids: cases
            .iter()
            .map(|case| {
                case["input_ids"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_i64().unwrap())
                    .collect()
            })
            .collect(),
        attention_mask: cases
            .iter()
            .map(|case| {
                case["attention_mask"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_i64().unwrap() != 0)
                    .collect()
            })
            .collect(),
    };
    let answers = engine
        .system_one(&State::Prepared(prepared), &questions)
        .unwrap();
    let close = |got: f64, expected: &serde_json::Value| {
        let expected = expected.as_f64().unwrap();
        assert!(
            (got - expected).abs() < 2e-4,
            "answer mismatch: Rust {got}, Julia {expected}"
        );
    };
    for (answer, case) in answers.iter().zip(cases) {
        let expected = &case["answer"];
        match answer {
            Answer::Choice(answer) => {
                assert_eq!(expected["type"], "choice");
                assert_eq!(answer.choice, expected["choice"].as_str().unwrap());
                close(answer.confidence, &expected["confidence"]);
                for (label, probability) in &answer.probabilities {
                    close(*probability, &expected["probabilities"][label]);
                }
            }
            Answer::Noul(answer) => {
                assert_eq!(expected["type"], "noul");
                close(answer.noul, &expected["noul"]);
            }
            Answer::Score(answer) => {
                assert_eq!(expected["type"], "score");
                close(answer.score, &expected["score"]);
                close(answer.confidence, &expected["confidence"]);
                for (i, (level, probability)) in
                    answer.legend.iter().zip(&answer.probabilities).enumerate()
                {
                    let key = i.to_string();
                    assert_eq!(level, expected["legend"][&key].as_str().unwrap());
                    close(*probability, &expected["probabilities"][&key]);
                }
            }
        }
    }
}
