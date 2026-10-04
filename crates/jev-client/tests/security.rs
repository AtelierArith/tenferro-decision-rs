//! Security regressions: credential redaction, fixed endpoint, and no remote
//! body leakage.

mod common;

use decision_core::State;

use jev_client::testing::MockTransport;
use jev_client::{
    EnvCredential, HttpResponse, JevError, MODELS_PATH, RecordingSleeper, SYSTEM_ONE_PATH,
    StaticCredential, SystemOneRequest, endpoint_url,
};

use common::{build_client, json_ok, noul_questions, noul_response};

const SENTINEL: &str = "SENTINEL-SECRET-VALUE-1234567890";

#[test]
fn credential_never_appears_in_requests_or_errors() {
    let mock = MockTransport::new();
    mock.push(json_ok(noul_response("jev-1.13.0", "is_urgent", 0.9)));
    let sleeper = RecordingSleeper::new();
    let client =
        jev_client::ClientBuilder::new(jev_client::PinnedModel::new("jev-1.13.0").unwrap())
            .credential(StaticCredential::new(SENTINEL).unwrap())
            .transport(mock.clone())
            .sleeper(sleeper.clone())
            .build()
            .unwrap();

    let state = State::Text("x".into());
    let questions = noul_questions();
    client
        .system_one(&SystemOneRequest::new(&state, &questions))
        .unwrap();

    let request = &mock.requests()[0];
    let debug = format!("{request:?}");
    assert!(!debug.contains(SENTINEL), "request Debug leaked the key");
    assert!(debug.contains("<redacted>"));
    assert!(!String::from_utf8_lossy(&request.body).contains(SENTINEL));
    assert_eq!(request.credential.expose_secret(), SENTINEL.as_bytes());
}

#[test]
fn remote_error_body_is_not_retained_or_displayed() {
    let mock = MockTransport::new();
    mock.push(HttpResponse::json(
        422,
        format!(r#"{{"error":"{SENTINEL}"}}"#),
    ));
    let sleeper = RecordingSleeper::new();
    let client = build_client(&mock, &sleeper);

    let state = State::Text("x".into());
    let questions = noul_questions();
    let error = client
        .system_one(&SystemOneRequest::new(&state, &questions))
        .unwrap_err();

    assert!(matches!(error, JevError::Api { status: 422, .. }));
    assert!(!format!("{error}").contains(SENTINEL));
    assert!(!format!("{error:?}").contains(SENTINEL));
}

#[test]
fn missing_env_credential_is_typed_and_redacted() {
    let mock = MockTransport::new();
    let sleeper = RecordingSleeper::new();
    let client =
        jev_client::ClientBuilder::new(jev_client::PinnedModel::new("jev-1.13.0").unwrap())
            .credential(EnvCredential::new("JEV_CLIENT_TEST_MISSING_ENV_VAR_XYZ"))
            .transport(mock)
            .sleeper(sleeper)
            .build()
            .unwrap();

    let state = State::Text("x".into());
    let questions = noul_questions();
    let error = client
        .system_one(&SystemOneRequest::new(&state, &questions))
        .unwrap_err();
    assert!(matches!(error, JevError::Credential { .. }));
}

#[test]
fn endpoint_is_fixed_and_paths_are_allowlisted() {
    assert_eq!(
        endpoint_url(SYSTEM_ONE_PATH).unwrap(),
        "https://api.typesafe.ai:443/v1/systemone"
    );
    assert_eq!(
        endpoint_url(MODELS_PATH).unwrap(),
        "https://api.typesafe.ai:443/v1/models"
    );
    assert!(endpoint_url("https://attacker.example/v1/systemone").is_err());
    assert!(endpoint_url("/v1/models/../../evil").is_err());
}

#[test]
fn proxy_env_does_not_change_the_endpoint() {
    // SAFETY: edition 2024 makes environment mutation unsafe because it can
    // race with other threads. This test only sets the variable around one
    // call and removes it immediately.
    unsafe {
        std::env::set_var("HTTPS_PROXY", "http://attacker.example:8080");
    }
    let result = endpoint_url(SYSTEM_ONE_PATH);
    unsafe {
        std::env::remove_var("HTTPS_PROXY");
    }
    assert_eq!(result.unwrap(), "https://api.typesafe.ai:443/v1/systemone");
}
