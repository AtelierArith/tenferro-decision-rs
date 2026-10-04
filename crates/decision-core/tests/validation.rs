use decision_core::prelude::*;
use decision_core::{DecisionEngine, MAX_CONTENT_DEPTH};

type Result<T, E> = std::result::Result<T, E>;

fn choice(instruction: &str, candidates: &[&str]) -> Question {
    Question::Choice(
        ChoiceQuestion::new(
            Content::string(instruction),
            candidates
                .iter()
                .map(|id| (id.to_string(), Content::string(format!("desc {id}"))))
                .collect(),
        )
        .unwrap(),
    )
}

#[test]
fn choice_requires_between_two_and_255_candidates() {
    let too_few = ChoiceQuestion::new(Content::string("q"), vec![("a".into(), Content::Null)]);
    assert!(too_few.is_err());

    let two = ChoiceQuestion::new(
        Content::string("q"),
        vec![("a".into(), Content::Null), ("b".into(), Content::Null)],
    );
    assert!(two.is_ok());
}

#[test]
fn score_requires_two_to_ten_levels() {
    assert!(ScoreQuestion::new(Content::string("q"), vec!["only".into()]).is_err());
    assert!(ScoreQuestion::new(Content::string("q"), vec!["".into(), "b".into()]).is_err());
    assert!(
        ScoreQuestion::new(
            Content::string("q"),
            (0..11).map(|i| i.to_string()).collect()
        )
        .is_err()
    );
    assert!(
        ScoreQuestion::new(
            Content::string("q"),
            (0..5).map(|i| i.to_string()).collect()
        )
        .is_ok()
    );
}

#[test]
fn noul_requires_at_least_one_criterion() {
    assert!(NoulQuestion::new(Content::string("q"), NoulCriteria::default()).is_err());
    let ok = NoulQuestion::new(
        Content::string("q"),
        NoulCriteria {
            truthy: Some(Content::string("yes")),
            falsy: None,
        },
    );
    assert!(ok.is_ok());
}

#[test]
fn duplicate_candidate_ids_rejected() {
    let err = ChoiceQuestion::new(
        Content::string("q"),
        vec![("a".into(), Content::Null), ("a".into(), Content::Null)],
    );
    assert!(err.is_err());
}

#[test]
fn empty_instructions_rejected() {
    assert!(
        ChoiceQuestion::new(
            Content::string(""),
            vec![("a".into(), Content::Null), ("b".into(), Content::Null)]
        )
        .is_err()
    );
    assert!(
        ChoiceQuestion::new(
            Content::Null,
            vec![("a".into(), Content::Null), ("b".into(), Content::Null)]
        )
        .is_err()
    );
}

#[test]
fn question_id_validation() {
    assert!(QuestionId::new("ok").is_ok());
    assert!(QuestionId::new("").is_err());
    assert!(QuestionId::new(" leading").is_err());
    assert!(QuestionId::new("trailing ").is_err());
    assert!(QuestionId::new("control\u{0007}").is_err());
    assert!(QuestionId::new("x".repeat(129)).is_err());
    assert!(QuestionId::new("x".repeat(128)).is_ok());
}

#[test]
fn question_set_requires_unique_ids_and_preserves_order() {
    let mut set = QuestionSet::new();
    set.push("first", choice("q1", &["a", "b"])).unwrap();
    set.push("second", choice("q2", &["c", "d"])).unwrap();

    assert!(set.push("first", choice("q3", &["e", "f"])).is_err());

    let ids: Vec<_> = set.questions().iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(ids, ["first", "second"]);
    assert!(set.get("second").is_some());
    assert!(set.get("missing").is_none());
    set.validate().unwrap();
}

#[test]
fn empty_question_set_is_invalid() {
    assert!(QuestionSet::new().validate().is_err());
}

#[test]
fn content_rejects_nan_and_infinity() {
    assert!(Content::float(f64::NAN).is_err());
    assert!(Content::float(f64::NEG_INFINITY).is_err());
    assert!(Content::Float(f64::NAN).validate().is_err());
}

#[test]
fn content_rejects_duplicate_object_keys() {
    let value = Content::Object(vec![
        ("k".into(), Content::Int(1)),
        ("k".into(), Content::Int(2)),
    ]);
    assert!(value.validate().is_err());
}

#[test]
fn content_rejects_excessive_depth() {
    let mut value = Content::Null;
    for _ in 0..(MAX_CONTENT_DEPTH + 2) {
        value = Content::Array(vec![value]);
    }
    assert!(value.validate().is_err());
}

#[test]
fn prepared_state_shape_consistency() {
    let good = PreparedState {
        input_ids: vec![vec![1, 2, 3]],
        attention_mask: vec![vec![true, true, false]],
    };
    good.validate().unwrap();

    let mismatched = PreparedState {
        input_ids: vec![vec![1, 2, 3]],
        attention_mask: vec![vec![true, true]],
    };
    assert!(mismatched.validate().is_err());

    let all_masked = PreparedState {
        input_ids: vec![vec![1]],
        attention_mask: vec![vec![false]],
    };
    assert!(all_masked.validate().is_err());
}

#[test]
fn state_validation_dispatches() {
    assert!(State::Text("hello".into()).validate().is_ok());
    assert!(State::Text(String::new()).validate().is_err());
    assert!(
        State::Json(Content::object([("a", Content::Int(1))]))
            .validate()
            .is_ok()
    );
}

#[test]
fn response_lookup_and_index() {
    let response = SystemOneResponse {
        model: "test-1".into(),
        answers: vec![("q1".into(), Answer::Noul(NoulAnswer { noul: 0.25 }))],
        usage: Usage {
            input_tokens: 3,
            output_tokens: 0,
        },
        request_id: Some("req".into()),
    };
    assert!(matches!(response.answer("q1"), Some(Answer::Noul(_))));
    assert!(response.answer("missing").is_none());
    assert!(matches!(response["q1"], Answer::Noul(_)));
}

/// A trivial engine proving the trait is implementable without any tensor
/// dependency.
struct EchoEngine;

impl DecisionEngine for EchoEngine {
    type Error = DecisionError;

    fn system_one(
        &mut self,
        state: &State,
        questions: &QuestionSet,
    ) -> Result<Vec<Answer>, Self::Error> {
        state.validate()?;
        questions.validate()?;
        Ok(questions
            .questions()
            .iter()
            .map(|_| Answer::Noul(NoulAnswer { noul: 0.5 }))
            .collect())
    }
}

#[test]
fn decision_engine_seam_is_implementable() {
    let mut engine = EchoEngine;
    let mut set = QuestionSet::new();
    set.push(
        "n1",
        Question::Noul(
            NoulQuestion::new(
                Content::string("is it so?"),
                NoulCriteria {
                    truthy: Some(Content::string("yes")),
                    falsy: Some(Content::string("no")),
                },
            )
            .unwrap(),
        ),
    )
    .unwrap();

    let answers = engine
        .system_one(&State::Text("context".into()), &set)
        .unwrap();
    assert_eq!(answers.len(), 1);
}
