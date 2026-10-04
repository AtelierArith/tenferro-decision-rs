//! Metadata-only observability.
//!
//! There is no logging framework dependency. Instead the client emits an
//! [`Event`] to an injected [`EventSink`]; the default is a no-op. Events carry
//! only safe metadata — never credentials, `state`, `instructions`, `criteria`,
//! bodies, candidate ids, or question ids
//! (`docs/agents/specs/docs/17_JEV_CLIENT_DESIGN.md` §10).

use std::time::Duration;

use decision_core::Usage;

/// Which operation produced an event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    /// A `POST /v1/systemone` call.
    SystemOne,
    /// A `GET /v1/models` call.
    ListModels,
}

/// A safe, metadata-only event.
#[derive(Clone, Debug, PartialEq)]
pub struct Event<'a> {
    /// The operation.
    pub operation: Operation,
    /// The model id requested.
    pub requested_model: &'a str,
    /// The model id the server returned, when known.
    pub returned_model: Option<&'a str>,
    /// Whether the requested model was a non-reproducible moving alias.
    pub moving_alias: bool,
    /// Number of questions in the request.
    pub question_count: usize,
    /// Serialized request body size in bytes.
    pub request_bytes: usize,
    /// Response body size in bytes.
    pub response_bytes: usize,
    /// HTTP status, when a response was received.
    pub status: Option<u16>,
    /// One-based attempt count.
    pub attempt: u32,
    /// The status that triggered a retry, when this attempt is retried.
    pub retried_status: Option<u16>,
    /// Upstream request id, when present.
    pub request_id: Option<&'a str>,
    /// Token usage, when present.
    pub usage: Option<Usage>,
    /// Elapsed time for the call.
    pub latency: Option<Duration>,
}

/// Receives metadata-only events.
pub trait EventSink: Send + Sync {
    /// Handle one event.
    fn on_event(&self, event: &Event<'_>);
}

/// The default sink: discards every event.
#[derive(Debug, Default)]
pub struct NoopEventSink;

impl EventSink for NoopEventSink {
    fn on_event(&self, _event: &Event<'_>) {}
}
