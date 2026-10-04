//! Lifecycle and concurrency behavior.

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use decision_core::State;

use jev_client::testing::MockTransport;
use jev_client::{
    with_client, ClientBuilder, HttpRequest, HttpResponse, JevError, PinnedModel, RecordingSleeper,
    StaticCredential, SystemOneRequest, Transport, TransportError,
};

use common::{build_client, json_ok, noul_questions, noul_response};

#[test]
fn close_stops_new_calls_and_is_idempotent() {
    let mock = MockTransport::new();
    mock.push(json_ok(noul_response("jev-1.13.0", "is_urgent", 0.9)));
    let sleeper = RecordingSleeper::new();
    let client = build_client(&mock, &sleeper);

    let state = State::Text("x".into());
    let questions = noul_questions();
    assert!(client.is_open());
    client
        .system_one(&SystemOneRequest::new(&state, &questions))
        .unwrap();

    client.close();
    assert!(!client.is_open());

    let error = client
        .system_one(&SystemOneRequest::new(&state, &questions))
        .unwrap_err();
    assert!(matches!(error, JevError::Closed));

    // Idempotent.
    client.close();
    client.close();
}

#[test]
fn with_client_returns_the_closure_value_and_closes() {
    let mock = MockTransport::new();
    mock.push(json_ok(noul_response("jev-1.13.0", "is_urgent", 0.9)));
    let sleeper = RecordingSleeper::new();
    let builder = ClientBuilder::new(PinnedModel::new("jev-1.13.0").unwrap())
        .credential(StaticCredential::new(common::TEST_SECRET).unwrap())
        .transport(mock)
        .sleeper(sleeper);

    let state = State::Text("x".into());
    let questions = noul_questions();
    let value = with_client(builder, |client| {
        assert!(client.is_open());
        let response = client
            .system_one(&SystemOneRequest::new(&state, &questions))
            .unwrap();
        response.model
    })
    .unwrap();
    assert_eq!(value, "jev-1.13.0");
}

#[derive(Clone, Debug, Default)]
struct ConcurrencyTransport {
    active: Arc<AtomicUsize>,
    max_seen: Arc<AtomicUsize>,
}

impl Transport for ConcurrencyTransport {
    fn execute(&self, _request: &HttpRequest) -> Result<HttpResponse, TransportError> {
        let now = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_seen.fetch_max(now, Ordering::SeqCst);
        thread::sleep(Duration::from_millis(80));
        self.active.fetch_sub(1, Ordering::SeqCst);
        Ok(HttpResponse::json(
            200,
            noul_response("jev-1.13.0", "is_urgent", 0.5),
        ))
    }
}

#[test]
fn max_inflight_bounds_concurrency() {
    let transport = ConcurrencyTransport::default();
    let builder = ClientBuilder::new(PinnedModel::new("jev-1.13.0").unwrap())
        .credential(StaticCredential::new(common::TEST_SECRET).unwrap())
        .transport(transport.clone())
        .sleeper(RecordingSleeper::new())
        .max_inflight(2);
    let client = builder.build().unwrap();

    let state = State::Text("x".into());
    let questions = noul_questions();
    let request = SystemOneRequest::new(&state, &questions);
    let outcomes: Vec<Result<_, JevError>> = thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| scope.spawn(|| client.system_one(&request)))
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    });
    assert!(outcomes.iter().all(Result::is_ok));
    assert!(
        transport.max_seen.load(Ordering::SeqCst) <= 2,
        "more than max_inflight calls ran concurrently"
    );
}
