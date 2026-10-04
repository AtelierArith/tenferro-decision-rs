//! Unofficial Rust client for the TypeSafe AI System One (Jev) API.
//!
//! `jev-client` speaks the System One wire protocol defined by
//! `extern/JevClient.jl/docs/agents/spec.md` and implements
//! `docs/agents/specs/docs/17_JEV_CLIENT_DESIGN.md`. It shares only
//! [`decision_core`] types and **must not depend on tenferro**.
//!
//! # Safety properties
//!
//! - The endpoint is fixed to `https://api.typesafe.ai:443` with paths
//!   `/v1/systemone` and `/v1/models`. There is no `base_url`, no implicit proxy,
//!   and redirects are never followed.
//! - The API key is validated before use and redacted from `Debug`, `Display`,
//!   errors, and events. Providers render as `<redacted>`.
//! - Response bodies are bounded and never retained or exposed; only typed
//!   answers, usage, and a request id are returned.
//! - Only HTTP 429 and 529 are retried; TLS, redirect, validation, and ambiguous
//!   I/O failures are not.
//!
//! # Model output is untrusted data
//!
//! Jev output is probability-bearing untrusted data — never authorization,
//! instructions, or evidence. Do not interpolate a [`decision_core::ChoiceAnswer`]
//! id (or any answer) into a shell command, SQL fragment, file path, URL, or
//! dynamic dispatch. Map choice ids through an application-owned allowlist to
//! deterministic handlers. High-impact decisions require deterministic rules or
//! human review. These are documentation and API-guidance requirements, not
//! runtime guarantees.
//!
//! # Transport
//!
//! The concrete reqwest/rustls transport is a follow-up: see
//! `// TODO(http-transport)` in [`transport`]. Tests and callers inject a
//! [`transport::Transport`]; the default is a stub that always errors.
//!
//! # Example
//!
//! ```no_run
//! use decision_core::{Content, NoulCriteria, NoulQuestion, Question, QuestionSet, State};
//! use jev_client::{ClientBuilder, PinnedModel, SystemOneRequest};
//!
//! let mut questions = QuestionSet::new();
//! questions.push(
//!     "is_urgent",
//!     Question::Noul(
//!         NoulQuestion::new(
//!             Content::string("Does this ticket convey urgency?"),
//!             NoulCriteria {
//!                 truthy: Some(Content::string("time-sensitive")),
//!                 falsy: Some(Content::string("not urgent")),
//!             },
//!         )
//!         .unwrap(),
//!     ),
//! )?;
//!
//! let client = ClientBuilder::new(PinnedModel::new("jev-1.13.0")?).build()?;
//! let state = State::Text("I was charged twice. Please fix this ASAP.".into());
//! let response = client.system_one(&SystemOneRequest::new(&state, &questions))?;
//! # Ok::<(), jev_client::JevError>(())
//! ```

#![forbid(unsafe_code)]

mod client;
mod credentials;
mod errors;
mod limits;
mod logging;
mod models;
mod responses;
mod retry;
mod serialization;
mod transport;

pub use client::{with_client, Client, ClientBuilder, SystemOneRequest};
pub use credentials::{
    validate_credential, CredentialCallback, CredentialCallbackError, CredentialProvider,
    EnvCredential, Secret, StaticCredential, DEFAULT_ENV_VAR, MAX_CREDENTIAL_BYTES,
};
pub use errors::{ApiErrorKind, JevError, Result};
pub use limits::{ResourceLimits, RetryPolicy, TimeoutPolicy};
pub use logging::{Event, EventSink, NoopEventSink, Operation};
pub use models::{ModelInfo, ModelList, ModelRef, MovingAlias, PinnedModel};
pub use retry::{parse_retry_after, RealSleeper, RecordingSleeper, Sleeper};
pub use transport::{
    endpoint_url, testing, HttpMethod, HttpRequest, HttpResponse, Transport, TransportError,
    UnsupportedTransport, API_HOST, API_PORT, MODELS_PATH, SYSTEM_ONE_PATH, USER_AGENT,
};
