//! End-to-end behavior through [`MockTransport`]; never touches the network.

mod common;

use std::time::Duration;

use decision_core::{Answer, Content, NoulAnswer, State};

use jev_client::testing::MockTransport;
use jev_client::{
    HttpResponse, JevError, RecordingSleeper, ResourceLimits, RetryPolicy, SystemOneRequest,
};

use common::{
    build_client, build_client_full, build_client_with, choice_questions, json_ok, noul_questions,
    noul_response,
};

#[test]
fn serializes_the_wire_request_shape() {
    let mock = MockTransport::new();
    mock.push(json_ok(noul_response("jev-1.13.0", "is_urgent", 0.9)));
    let sleeper = RecordingSleeper::new();
    let client = build_client(&mock, &sleeper);

    let state = State::Json(Content::object([
        ("z", Content::Int(1)),
        ("a", Content::Int(2)),
    ]));
    let questions = noul_questions();
    client
        .system_one(&SystemOneRequest::new(&state, &questions))
        .unwrap();

    let requests = mock.requests();
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert_eq!(request.method.as_str(), "POST");
    assert_eq!(request.path, "/v1/systemone");
    assert_eq!(request.header("accept"), Some("application/json"));
    assert_eq!(request.header("accept-encoding"), Some("identity"));
    assert!(
        request
            .header("content-type")
            .unwrap()
            .starts_with("application/json")
    );
    assert!(request.header("authorization").is_none());
    assert_eq!(
        request.credential.expose_secret(),
        common::TEST_SECRET.as_bytes()
    );

    // Insertion order is preserved for the top-level and nested objects.
    let body = String::from_utf8(request.body.clone()).unwrap();
    assert_eq!(
        body,
        r#"{"state":{"z":1,"a":2},"model":"jev-1.13.0","questions":{"is_urgent":{"type":"noul","instructions":"Is this urgent?","criteria":{"true":"time-sensitive","false":"not urgent"}}}}"#
    );
}

#[test]
fn successful_system_one_returns_typed_answers() {
    let mock = MockTransport::new();
    mock.push(json_ok(noul_response("jev-1.13.0", "is_urgent", 0.9)));
    let sleeper = RecordingSleeper::new();
    let client = build_client(&mock, &sleeper);

    let state = State::Text("I was charged twice.".into());
    let questions = noul_questions();
    let response = client
        .system_one(&SystemOneRequest::new(&state, &questions))
        .unwrap();

    assert_eq!(response.model, "jev-1.13.0");
    assert_eq!(response.usage.input_tokens, 3);
    assert_eq!(response.usage.output_tokens, 5);
    assert_eq!(
        response.answer("is_urgent"),
        Some(&Answer::Noul(NoulAnswer { noul: 0.9 }))
    );
}

#[test]
fn retries_on_429_then_succeeds() {
    let mock = MockTransport::new();
    mock.push(HttpResponse::json(429, b"{}".to_vec()));
    mock.push(json_ok(noul_response("jev-1.13.0", "is_urgent", 0.4)));
    let sleeper = RecordingSleeper::new();
    let client = build_client(&mock, &sleeper);

    let state = State::Text("x".into());
    let questions = noul_questions();
    client
        .system_one(&SystemOneRequest::new(&state, &questions))
        .unwrap();

    assert_eq!(mock.request_count(), 2);
    assert_eq!(sleeper.count(), 1);
}

#[test]
fn retries_on_529_then_succeeds() {
    let mock = MockTransport::new();
    mock.push(HttpResponse::json(529, b"{}".to_vec()));
    mock.push(json_ok(noul_response("jev-1.13.0", "is_urgent", 0.4)));
    let sleeper = RecordingSleeper::new();
    let client = build_client(&mock, &sleeper);

    let state = State::Text("x".into());
    let questions = noul_questions();
    client
        .system_one(&SystemOneRequest::new(&state, &questions))
        .unwrap();

    assert_eq!(mock.request_count(), 2);
    assert_eq!(sleeper.count(), 1);
}

#[test]
fn does_not_retry_other_statuses() {
    let mock = MockTransport::new();
    mock.push(HttpResponse::json(500, b"{}".to_vec()));
    let sleeper = RecordingSleeper::new();
    let client = build_client(&mock, &sleeper);

    let state = State::Text("x".into());
    let questions = noul_questions();
    let error = client
        .system_one(&SystemOneRequest::new(&state, &questions))
        .unwrap_err();

    assert!(matches!(error, JevError::Api { status: 500, .. }));
    assert_eq!(mock.request_count(), 1);
    assert_eq!(sleeper.count(), 0);
}

#[test]
fn retry_budget_exhaustion_is_typed() {
    let mock = MockTransport::new();
    // A server-requested delay far larger than the total budget.
    mock.push(HttpResponse::json(429, b"{}".to_vec()).with_header("Retry-After", "100"));
    let sleeper = RecordingSleeper::new();
    let client = build_client_full(
        &mock,
        &sleeper,
        ResourceLimits::default(),
        RetryPolicy {
            max_retries: 2,
            initial_delay: Duration::from_millis(1),
            max_delay: Duration::from_secs(1),
            total_budget: Duration::from_secs(1),
        },
    );

    let state = State::Text("x".into());
    let questions = noul_questions();
    let error = client
        .system_one(&SystemOneRequest::new(&state, &questions))
        .unwrap_err();
    assert!(matches!(error, JevError::RetryBudgetExceeded { .. }));
    assert_eq!(sleeper.count(), 0);
}

#[test]
fn retries_exhausted_returns_rate_limit() {
    let mock = MockTransport::new();
    mock.push(HttpResponse::json(429, b"{}".to_vec()));
    mock.push(HttpResponse::json(429, b"{}".to_vec()));
    mock.push(HttpResponse::json(429, b"{}".to_vec()));
    let sleeper = RecordingSleeper::new();
    let client = build_client(&mock, &sleeper);

    let state = State::Text("x".into());
    let questions = noul_questions();
    let error = client
        .system_one(&SystemOneRequest::new(&state, &questions))
        .unwrap_err();

    assert!(matches!(error, JevError::Api { status: 429, .. }));
    // One initial attempt plus `max_retries` (2) retries.
    assert_eq!(mock.request_count(), 3);
}

#[test]
fn rejects_redirects_without_following() {
    let mock = MockTransport::new();
    mock.push(
        HttpResponse::json(302, b"".to_vec())
            .with_header("Location", "https://attacker.example/steal"),
    );
    let sleeper = RecordingSleeper::new();
    let client = build_client(&mock, &sleeper);

    let state = State::Text("x".into());
    let questions = noul_questions();
    let error = client
        .system_one(&SystemOneRequest::new(&state, &questions))
        .unwrap_err();

    assert!(matches!(error, JevError::Redirect { status: 302 }));
    assert_eq!(mock.request_count(), 1);
}

#[test]
fn rejects_oversized_response_body() {
    let limits = ResourceLimits {
        max_response_bytes: 32,
        ..ResourceLimits::default()
    };
    let mock = MockTransport::new();
    mock.push(json_ok(noul_response("jev-1.13.0", "is_urgent", 0.9)));
    let sleeper = RecordingSleeper::new();
    let client = build_client_with(&mock, &sleeper, limits);

    let state = State::Text("x".into());
    let questions = noul_questions();
    let error = client
        .system_one(&SystemOneRequest::new(&state, &questions))
        .unwrap_err();
    assert!(matches!(error, JevError::ResponseTooLarge { .. }));
}

#[test]
fn rejects_oversized_content_length_before_reading() {
    let limits = ResourceLimits {
        max_response_bytes: 32,
        ..ResourceLimits::default()
    };
    let mock = MockTransport::new();
    mock.push(HttpResponse::json(200, b"{}".to_vec()).with_header("Content-Length", "1000000"));
    let sleeper = RecordingSleeper::new();
    let client = build_client_with(&mock, &sleeper, limits);

    let state = State::Text("x".into());
    let questions = noul_questions();
    let error = client
        .system_one(&SystemOneRequest::new(&state, &questions))
        .unwrap_err();
    assert!(matches!(error, JevError::ResponseTooLarge { .. }));
}

#[test]
fn rejects_duplicate_json_keys() {
    let body = br#"{"model":"m","model":"m2","answers":{"is_urgent":{"type":"noul","noul":0.5}},"usage":{"input_tokens":1,"output_tokens":1}}"#;
    let mock = MockTransport::new();
    mock.push(json_ok(body.to_vec()));
    let sleeper = RecordingSleeper::new();
    let client = build_client(&mock, &sleeper);

    let state = State::Text("x".into());
    let questions = noul_questions();
    let error = client
        .system_one(&SystemOneRequest::new(&state, &questions))
        .unwrap_err();
    assert!(matches!(error, JevError::MalformedJson { .. }));
}

#[test]
fn rejects_malformed_json() {
    let mock = MockTransport::new();
    mock.push(json_ok(b"{\"model\":".to_vec()));
    let sleeper = RecordingSleeper::new();
    let client = build_client(&mock, &sleeper);

    let state = State::Text("x".into());
    let questions = noul_questions();
    let error = client
        .system_one(&SystemOneRequest::new(&state, &questions))
        .unwrap_err();
    assert!(matches!(error, JevError::MalformedJson { .. }));
}

#[test]
fn rejects_non_json_content_type() {
    let mock = MockTransport::new();
    mock.push(HttpResponse::new(
        200,
        vec![("content-type".into(), "text/html".into())],
        b"<html></html>".to_vec(),
    ));
    let sleeper = RecordingSleeper::new();
    let client = build_client(&mock, &sleeper);

    let state = State::Text("x".into());
    let questions = noul_questions();
    let error = client
        .system_one(&SystemOneRequest::new(&state, &questions))
        .unwrap_err();
    assert!(matches!(error, JevError::UnexpectedContentType));
}

#[test]
fn validates_choice_cross_fields() {
    let mock = MockTransport::new();
    // Keys match but the sum is wrong.
    mock.push(json_ok(
        br#"{"model":"m","answers":{"department":{"type":"choice","choice":"billing","probabilities":{"billing":0.9,"sales":0.9},"confidence":0.9}},"usage":{"input_tokens":1,"output_tokens":1}}"#
            .to_vec(),
    ));
    let sleeper = RecordingSleeper::new();
    let client = build_client(&mock, &sleeper);

    let state = State::Text("x".into());
    let questions = choice_questions();
    let error = client
        .system_one(&SystemOneRequest::new(&state, &questions))
        .unwrap_err();
    assert!(matches!(error, JevError::ResponseValidation { .. }));
}
