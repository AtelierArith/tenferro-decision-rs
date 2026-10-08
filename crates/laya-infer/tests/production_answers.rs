//! End-to-end production parity with the upstream bundled email example.
//! Skips when the pinned checkpoint or Julia capture is absent.
use decision_core::{
    Answer, ChoiceQuestion, Content, DecisionEngine, NoulCriteria, NoulQuestion, Question,
    QuestionSet, ScoreQuestion, State,
};
use hf_fetch::{CheckpointSpec, Hub};
use laya_infer::{
    agent::LayaEngine,
    model::{TensorCache, forward_tenferro_cached},
    prompt::build_sequence,
};
use serde_json::Value;
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;

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
fn close(got: f64, expected: &Value, tolerance: f64) {
    let expected = expected.as_f64().unwrap();
    assert!(
        (got - expected).abs() <= tolerance,
        "Rust {got}, Julia {expected}"
    );
}

#[test]
fn bundled_questions_match_julia_collate_predict_and_action() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/laya-real/answers.json");
    let Ok(text) = std::fs::read_to_string(path) else {
        eprintln!("skipping: Julia answer capture absent");
        return;
    };
    let reference: Value = serde_json::from_str(&text).unwrap();
    let mut spec = CheckpointSpec::laya();
    spec.revision = "7b928d828b7b0e022f929d9bd2e44165aa270148".into();
    let mut hub = Hub::from_env();
    hub.offline = true;
    let Ok(dir) = hub.resolve(&spec) else {
        eprintln!("skipping: pinned production Laya checkpoint absent");
        return;
    };
    assert_eq!(reference["checkpoint_revision"], spec.revision);
    let state = State::Json(Content::object(
        reference["state_entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|pair| {
                (
                    pair[0].as_str().unwrap(),
                    Content::string(pair[1].as_str().unwrap()),
                )
            }),
    ));
    let mut questions = QuestionSet::new();
    for (i, id) in reference["question_ids"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let id = id.as_str().unwrap();
        let q = &reference["questions"][id];
        let instructions = Content::string(q["instructions"].as_str().unwrap());
        let question = match q["type"].as_str().unwrap() {
            "choice" => Question::Choice(
                ChoiceQuestion::new(
                    instructions,
                    reference["criteria_order"][i]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|label| {
                            let label = label.as_str().unwrap();
                            (
                                label.to_string(),
                                Content::string(q["criteria"][label].as_str().unwrap()),
                            )
                        })
                        .collect(),
                )
                .unwrap(),
            ),
            "score" => Question::Score(
                ScoreQuestion::new(
                    instructions,
                    q["criteria"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_str().unwrap().to_string())
                        .collect(),
                )
                .unwrap(),
            ),
            "noul" => Question::Noul(
                NoulQuestion::new(
                    instructions,
                    NoulCriteria {
                        truthy: Some(Content::Null),
                        falsy: None,
                    },
                )
                .unwrap(),
            ),
            other => panic!("unexpected type {other}"),
        };
        questions.push(id, question).unwrap();
    }
    let mut engine = LayaEngine::load(dir).unwrap();
    for ((_, question), item) in questions
        .questions()
        .iter()
        .zip(reference["items"].as_array().unwrap())
    {
        let (ids, markers) = build_sequence(
            engine.tokenizer(),
            &state,
            question,
            engine.agent().max_len,
            engine.agent().head_max_len,
        )
        .unwrap();
        assert_eq!(ids, integers(&item["ids"]));
        assert_eq!(
            markers.iter().map(|p| *p as i64).collect::<Vec<_>>(),
            integers(&item["markers"])
        );
    }
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let b = &reference["batch"];
    let (logits, action) = runtime
        .with_eager_session(|session| {
            forward_tenferro_cached(
                session,
                &mut TensorCache::new(),
                engine.encoder(),
                engine.agent(),
                engine.weights(),
                &integers(&b["input_ids"]),
                &booleans(&b["attention_mask"]),
                &integers(&b["marker_pos"]),
                &booleans(&b["marker_mask"]),
                &integers(&b["qtype"]),
            )
        })
        .unwrap()
        .unwrap();
    for (actual, key) in [(&logits, "logits"), (&action, "action")] {
        let expected = reference[key].as_array().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(expected) {
            close(
                f64::from(*got),
                want,
                2e-3 + want.as_f64().unwrap().abs() * 2e-5,
            );
        }
    }
    let decisions = engine.decide(&state, &questions).unwrap();
    let answers = engine.system_one(&state, &questions).unwrap();
    assert_eq!(
        answers,
        decisions
            .iter()
            .map(|d| d.answer.clone())
            .collect::<Vec<_>>()
    );
    for ((id, _), decision) in questions.questions().iter().zip(decisions) {
        let expected = &reference["prediction"]["answers"][id.as_str()];
        close(
            decision.action_probability,
            &expected["action"]["act_probability"],
            2e-4,
        );
        match decision.answer {
            Answer::Choice(answer) => {
                assert_eq!(answer.choice, expected["choice"].as_str().unwrap());
                close(answer.confidence, &expected["confidence"], 2e-4);
                for (label, p) in answer.probabilities {
                    close(p, &expected["probabilities"][label], 2e-4);
                }
            }
            Answer::Score(answer) => {
                close(answer.score, &expected["score"], 2e-4);
                close(answer.confidence, &expected["confidence"], 2e-4);
                for (i, (label, p)) in answer.legend.iter().zip(answer.probabilities).enumerate() {
                    assert_eq!(label, expected["legend"][i.to_string()].as_str().unwrap());
                    close(p, &expected["probabilities"][i.to_string()], 2e-4);
                }
            }
            Answer::Noul(answer) => close(answer.noul, &expected["noul"], 2e-4),
        }
    }
}
