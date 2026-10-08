//! Golden prompts/token ids captured from original Jeff, and optional Julia answers.
use decision_core::{
    Answer, ChoiceQuestion, Content, DecisionEngine, NoulCriteria, NoulQuestion, Question,
    QuestionSet, ScoreQuestion, State,
};
use hf_fetch::{CheckpointSpec, Hub};
use jeff_infer::{engine::JeffEngine, tokenizer::JeffTokenizer};
use serde_json::Value;

fn content(value: &Value) -> Content {
    match value {
        Value::Null => Content::Null,
        Value::Bool(v) => Content::Bool(*v),
        Value::Number(v) if v.is_i64() => Content::Int(v.as_i64().unwrap()),
        Value::Number(v) => Content::Float(v.as_f64().unwrap()),
        Value::String(v) => Content::string(v),
        Value::Array(v) => Content::Array(v.iter().map(content).collect()),
        Value::Object(v) => Content::object(v.iter().map(|(k, v)| (k, content(v)))),
    }
}
fn inputs(case: &Value) -> (State, QuestionSet) {
    let state = if let Some(text) = case["state"].as_str() {
        State::Text(text.into())
    } else {
        State::Json(Content::object(
            case["state_order"].as_array().unwrap().iter().map(|key| {
                let key = key.as_str().unwrap();
                (key, content(&case["state"][key]))
            }),
        ))
    };
    let q = &case["question"];
    let instructions = content(&q["instructions"]);
    let question = match q["type"].as_str().unwrap() {
        "choice" => Question::Choice(
            ChoiceQuestion::new(
                instructions,
                case["criteria_order"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|key| {
                        let key = key.as_str().unwrap();
                        (key.into(), content(&q["criteria"][key]))
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
                    .map(|v| v.as_str().unwrap().into())
                    .collect(),
            )
            .unwrap(),
        ),
        "noul" => Question::Noul(
            NoulQuestion::new(
                instructions,
                NoulCriteria {
                    truthy: q["criteria"].get("true").map(content),
                    falsy: q["criteria"].get("false").map(content),
                },
            )
            .unwrap(),
        ),
        other => panic!("unexpected question type {other}"),
    };
    let mut questions = QuestionSet::new();
    questions.push("q", question).unwrap();
    (state, questions)
}
fn available() -> Option<(std::path::PathBuf, Value)> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/jeff-real/tokenizer.json");
    let text = std::fs::read_to_string(path).ok()?;
    let reference: Value = serde_json::from_str(&text).unwrap();
    let mut hub = Hub::from_env();
    hub.offline = true;
    let dir = hub.resolve(&CheckpointSpec::jeff()).ok()?;
    assert_eq!(
        reference["checkpoint_revision"],
        CheckpointSpec::jeff().revision
    );
    Some((dir, reference))
}
#[test]
fn prompts_and_token_ids_match_original_jeff() {
    let Some((dir, reference)) = available() else {
        eprintln!("skipping: checkpoint or tokenizer capture absent");
        return;
    };
    let tokenizer = JeffTokenizer::from_directory(dir).unwrap();
    for sample in reference["samples"].as_array().unwrap() {
        let expected: Vec<i64> = sample["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        assert_eq!(
            tokenizer.encode(sample["text"].as_str().unwrap()).unwrap(),
            expected
        );
    }
    for case in reference["cases"].as_array().unwrap() {
        let (state, questions) = inputs(case);
        assert_eq!(
            tokenizer
                .render(&state, &questions.questions()[0].1)
                .unwrap(),
            case["prompt"].as_str().unwrap()
        );
        let prepared = tokenizer.prepare(&state, &questions).unwrap();
        let expected: Vec<i64> = case["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        assert_eq!(prepared.input_ids, vec![expected]);
    }
}
#[test]
#[ignore = "loads the production Jeff checkpoint; run with --release -- --ignored"]
fn text_and_json_answers_match_julia_on_original_token_ids() {
    let Some((dir, reference)) = available() else {
        eprintln!("skipping: checkpoint or tokenizer capture absent");
        return;
    };
    let mut engine = JeffEngine::load(dir).unwrap();
    let close = |got: f64, want: &Value| assert!((got - want.as_f64().unwrap()).abs() < 2e-4);
    for case in reference["cases"].as_array().unwrap() {
        let (state, questions) = inputs(case);
        let expected = &case["julia_answer"];
        assert!(
            !expected.is_null(),
            "generate Julia answers for the original token ids"
        );
        let answers = engine.system_one(&state, &questions).unwrap();
        match &answers[0] {
            Answer::Choice(a) => {
                assert_eq!(a.choice, expected["choice"].as_str().unwrap());
                close(a.confidence, &expected["confidence"]);
                for (label, p) in &a.probabilities {
                    close(*p, &expected["probabilities"][label]);
                }
            }
            Answer::Score(a) => {
                close(a.score, &expected["score"]);
                close(a.confidence, &expected["confidence"]);
                for (i, p) in a.probabilities.iter().enumerate() {
                    close(*p, &expected["probabilities"][i.to_string()]);
                }
            }
            Answer::Noul(a) => close(a.noul, &expected["noul"]),
        }
    }
}
