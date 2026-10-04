//! Shared helpers for the integration tests.
#![allow(dead_code)]

use decision_core::{
    ChoiceQuestion, Content, NoulCriteria, NoulQuestion, Question, QuestionSet, ScoreQuestion,
};

use jev_client::testing::MockTransport;
use jev_client::{
    Client, ClientBuilder, HttpResponse, PinnedModel, RecordingSleeper, ResourceLimits,
    RetryPolicy, StaticCredential,
};

pub const TEST_SECRET: &str = "test-key-abc-123";

pub fn noul_questions() -> QuestionSet {
    let mut questions = QuestionSet::new();
    questions
        .push(
            "is_urgent",
            Question::Noul(
                NoulQuestion::new(
                    Content::string("Is this urgent?"),
                    NoulCriteria {
                        truthy: Some(Content::string("time-sensitive")),
                        falsy: Some(Content::string("not urgent")),
                    },
                )
                .unwrap(),
            ),
        )
        .unwrap();
    questions
}

pub fn choice_questions() -> QuestionSet {
    let mut questions = QuestionSet::new();
    questions
        .push(
            "department",
            Question::Choice(
                ChoiceQuestion::new(
                    Content::string("Which team?"),
                    vec![
                        ("billing".into(), Content::string("Payments")),
                        ("sales".into(), Content::string("Pricing")),
                    ],
                )
                .unwrap(),
            ),
        )
        .unwrap();
    questions
}

pub fn score_questions() -> QuestionSet {
    let mut questions = QuestionSet::new();
    questions
        .push(
            "frustration",
            Question::Score(
                ScoreQuestion::new(
                    Content::string("How frustrated?"),
                    vec!["Calm".into(), "Angry".into()],
                )
                .unwrap(),
            ),
        )
        .unwrap();
    questions
}

pub fn noul_response(model: &str, id: &str, noul: f64) -> String {
    format!(
        r#"{{"model":"{model}","answers":{{"{id}":{{"type":"noul","noul":{noul}}}}},"usage":{{"input_tokens":3,"output_tokens":5}}}}"#
    )
}

pub fn build_client(mock: &MockTransport, sleeper: &RecordingSleeper) -> Client {
    build_client_with(mock, sleeper, ResourceLimits::default())
}

pub fn build_client_with(
    mock: &MockTransport,
    sleeper: &RecordingSleeper,
    limits: ResourceLimits,
) -> Client {
    build_client_full(mock, sleeper, limits, RetryPolicy::default())
}

pub fn build_client_full(
    mock: &MockTransport,
    sleeper: &RecordingSleeper,
    limits: ResourceLimits,
    retry: RetryPolicy,
) -> Client {
    ClientBuilder::new(PinnedModel::new("jev-1.13.0").unwrap())
        .credential(StaticCredential::new(TEST_SECRET).unwrap())
        .transport(mock.clone())
        .sleeper(sleeper.clone())
        .limits(limits)
        .retry(retry)
        .build()
        .unwrap()
}

pub fn json_ok(body: impl Into<Vec<u8>>) -> HttpResponse {
    HttpResponse::json(200, body)
}
