//! The injectable HTTP transport seam and the fixed endpoint policy.
//!
//! The concrete reqwest/rustls transport is a documented follow-up
//! (`// TODO(http-transport)`); this module defines the trait and the request and
//! response shapes so that tests can drive the client through
//! [`testing::MockTransport`] without touching the network. Requests carry only
//! a *path* — never a host, scheme, or full URL — so the client cannot be
//! pointed at another origin (`docs/agents/specs/docs/17_JEV_CLIENT_DESIGN.md`
//! §3–§4).

use std::fmt;
use std::time::Duration;

use crate::credentials::Secret;
use crate::errors::{JevError, Result};

/// The only permitted API host.
pub const API_HOST: &str = "api.typesafe.ai";
/// The only permitted API port.
pub const API_PORT: u16 = 443;
/// The System One (POST) path.
pub const SYSTEM_ONE_PATH: &str = "/v1/systemone";
/// The model-list (GET) path.
pub const MODELS_PATH: &str = "/v1/models";
/// The `User-Agent` sent with every request.
pub const USER_AGENT: &str = concat!("jev-client-rs/", env!("CARGO_PKG_VERSION"));

/// Resolve a permitted request path to the one fixed origin.
///
/// This is the *only* place a URL is formed. Any path outside the fixed policy
/// is rejected, and the result never contains userinfo, a query, or a fragment.
pub fn endpoint_url(path: &str) -> Result<String> {
    if path != SYSTEM_ONE_PATH && path != MODELS_PATH {
        return Err(JevError::Endpoint {
            message: "unsupported endpoint path".into(),
        });
    }
    Ok(format!("https://{API_HOST}:{API_PORT}{path}"))
}

/// HTTP method used by a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HttpMethod {
    /// `GET`.
    Get,
    /// `POST`.
    Post,
}

impl HttpMethod {
    /// The wire method name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
        }
    }
}

/// A bounded request handed to a [`Transport`].
///
/// The `Authorization` header is not part of `headers`; the credential travels
/// separately as a redacted [`Secret`] so it is never captured by header
/// logging or a recorded mock request. The request contains no host or URL.
#[derive(Clone)]
pub struct HttpRequest {
    /// HTTP method.
    pub method: HttpMethod,
    /// Permitted path, one of [`SYSTEM_ONE_PATH`] or [`MODELS_PATH`].
    pub path: String,
    /// Fixed headers (never `Authorization`, `Host`, cookies, or compression).
    pub headers: Vec<(String, String)>,
    /// The bearer credential for this attempt.
    pub credential: Secret,
    /// Request body; empty for `GET`.
    pub body: Vec<u8>,
    /// Connection timeout passed to the transport.
    pub connect_timeout: Duration,
    /// First-byte timeout passed to the transport.
    pub first_byte_timeout: Duration,
    /// Whole-attempt timeout passed to the transport.
    pub attempt_timeout: Duration,
    /// Response byte cap enforced by the transport.
    pub max_response_bytes: usize,
}

impl HttpRequest {
    /// Look up a response/request header case-insensitively.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

impl fmt::Debug for HttpRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpRequest")
            .field("method", &self.method)
            .field("path", &self.path)
            .field("headers", &self.headers)
            .field("credential", &self.credential)
            .field("body_len", &self.body.len())
            .finish()
    }
}

/// A response returned by a [`Transport`].
#[derive(Clone, PartialEq, Eq)]
pub struct HttpResponse {
    /// HTTP status code.
    pub status: u16,
    /// Response headers.
    pub headers: Vec<(String, String)>,
    /// Response body bytes. Never exposed beyond parsing.
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// Build a response.
    pub fn new(status: u16, headers: Vec<(String, String)>, body: Vec<u8>) -> Self {
        Self {
            status,
            headers,
            body,
        }
    }

    /// Build a JSON response with a `Content-Type` header.
    pub fn json(status: u16, body: impl Into<Vec<u8>>) -> Self {
        Self {
            status,
            headers: vec![(
                "content-type".into(),
                "application/json; charset=utf-8".into(),
            )],
            body: body.into(),
        }
    }

    /// Build a response with a `Retry-After` header.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Look up a header case-insensitively.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

impl fmt::Debug for HttpResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpResponse")
            .field("status", &self.status)
            .field("headers", &self.headers)
            .field("body_len", &self.body.len())
            .finish()
    }
}

/// A transport-layer failure.
///
/// No variant carries a response body. TLS and I/O failures are never retried
/// (`docs/agents/specs/docs/17_JEV_CLIENT_DESIGN.md` §6).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransportError {
    /// Connection could not be established.
    Connect(String),
    /// TLS certificate or hostname verification failed.
    Tls(String),
    /// A timeout elapsed.
    Timeout(String),
    /// An ambiguous I/O failure; the body may already have been sent.
    Io(String),
    /// The endpoint policy was violated before any network activity.
    Endpoint(String),
    /// No transport is configured (the concrete transport is a follow-up).
    Unsupported(String),
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Connect(message) => write!(f, "connect error: {message}"),
            Self::Tls(message) => write!(f, "TLS error: {message}"),
            Self::Timeout(message) => write!(f, "timeout: {message}"),
            Self::Io(message) => write!(f, "I/O error: {message}"),
            Self::Endpoint(message) => write!(f, "endpoint error: {message}"),
            Self::Unsupported(message) => write!(f, "unsupported transport: {message}"),
        }
    }
}

impl std::error::Error for TransportError {}

/// The injectable transport seam.
///
/// Implementations must send requests only to the fixed
/// [`endpoint_url`] origin, must disable redirect following and implicit
/// proxies, must verify TLS, and must enforce [`HttpRequest::max_response_bytes`].
pub trait Transport: Send + Sync {
    /// Execute a single attempt. Redirects and retries are handled by the
    /// client, not here.
    fn execute(&self, request: &HttpRequest) -> std::result::Result<HttpResponse, TransportError>;
}

/// The default transport when none is configured.
///
/// The concrete reqwest/rustls transport is not part of this task
/// (`// TODO(http-transport)`). Building a client without an explicit transport
/// yields this stub, so production code must inject one.
#[derive(Debug, Default)]
pub struct UnsupportedTransport;

impl Transport for UnsupportedTransport {
    fn execute(&self, _request: &HttpRequest) -> std::result::Result<HttpResponse, TransportError> {
        Err(TransportError::Unsupported(
            "no HTTP transport is configured; the concrete reqwest/rustls transport is a follow-up"
                .into(),
        ))
    }
}

/// Test-only transport helpers. Never used in production code.
pub mod testing {
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    use super::{HttpRequest, HttpResponse, Transport, TransportError};

    /// A transport that records requests and returns queued responses.
    ///
    /// Each call pops the next queued result; running out of queued responses
    /// is a [`TransportError::Connect`]. Cloning a `MockTransport` shares the
    /// same queue and request log, so a test can move one clone into a
    /// [`Client`](crate::Client) and keep another to inspect. Recorded requests
    /// are safe to inspect because [`HttpRequest`] redacts its credential.
    #[derive(Clone, Debug, Default)]
    pub struct MockTransport {
        inner: Arc<MockInner>,
    }

    #[derive(Debug, Default)]
    struct MockInner {
        requests: Mutex<Vec<HttpRequest>>,
        results: Mutex<VecDeque<std::result::Result<HttpResponse, TransportError>>>,
    }

    impl MockTransport {
        /// An empty mock.
        pub fn new() -> Self {
            Self::default()
        }

        /// A mock pre-loaded with responses.
        pub fn with_responses(responses: Vec<HttpResponse>) -> Self {
            let mock = Self::new();
            for response in responses {
                mock.push(response);
            }
            mock
        }

        /// Queue a response.
        pub fn push(&self, response: HttpResponse) {
            self.push_result(Ok(response));
        }

        /// Queue an arbitrary result.
        pub fn push_result(&self, result: std::result::Result<HttpResponse, TransportError>) {
            self.inner
                .results
                .lock()
                .expect("mock transport queue poisoned")
                .push_back(result);
        }

        /// Cloned copies of every request seen so far.
        pub fn requests(&self) -> Vec<HttpRequest> {
            self.inner
                .requests
                .lock()
                .expect("mock transport request log poisoned")
                .clone()
        }

        /// Number of requests seen so far.
        pub fn request_count(&self) -> usize {
            self.inner
                .requests
                .lock()
                .expect("mock transport request log poisoned")
                .len()
        }
    }

    impl Transport for MockTransport {
        fn execute(
            &self,
            request: &HttpRequest,
        ) -> std::result::Result<HttpResponse, TransportError> {
            self.inner
                .requests
                .lock()
                .expect("mock transport request log poisoned")
                .push(request.clone());
            self.inner
                .results
                .lock()
                .expect("mock transport queue poisoned")
                .pop_front()
                .unwrap_or(Err(TransportError::Connect(
                    "mock transport has no queued response".into(),
                )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_is_fixed() {
        assert_eq!(
            endpoint_url(SYSTEM_ONE_PATH).unwrap(),
            "https://api.typesafe.ai:443/v1/systemone"
        );
        assert_eq!(
            endpoint_url(MODELS_PATH).unwrap(),
            "https://api.typesafe.ai:443/v1/models"
        );
    }

    #[test]
    fn endpoint_rejects_other_paths() {
        // A caller cannot smuggle in a host, scheme, or alternate path.
        assert!(endpoint_url("https://attacker.example/v1/systemone").is_err());
        assert!(endpoint_url("//attacker.example/x").is_err());
        assert!(endpoint_url("/v1/systemone/../other").is_err());
        assert!(endpoint_url("").is_err());
    }

    #[test]
    fn endpoint_never_contains_userinfo_query_or_fragment() {
        for path in [SYSTEM_ONE_PATH, MODELS_PATH] {
            let url = endpoint_url(path).unwrap();
            assert!(url.starts_with("https://api.typesafe.ai:443/"));
            assert!(!url.contains('@'));
            assert!(!url.contains('?'));
            assert!(!url.contains('#'));
        }
    }
}
