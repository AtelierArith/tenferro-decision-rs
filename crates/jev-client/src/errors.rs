//! The typed error hierarchy for [`crate`].
//!
//! Errors carry only safe context — a classification message, HTTP status,
//! upstream request id, `Retry-After`, a field path, and an attempt count. Raw
//! request/response bodies and credentials are never stored or formatted
//! (`docs/agents/specs/docs/07_SECURITY_LICENSE.md` §3).

use thiserror::Error;

use decision_core::DecisionError;

/// Convenience result alias for fallible client operations.
pub type Result<T> = std::result::Result<T, JevError>;

/// The upstream API error class, derived from the HTTP status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApiErrorKind {
    /// HTTP 401 — the API key is missing or invalid.
    Authentication,
    /// HTTP 403 — the API key is not permitted.
    PermissionDenied,
    /// HTTP 404 — the endpoint was not found.
    NotFound,
    /// HTTP 422 — the request failed remote validation.
    RemoteValidation,
    /// HTTP 429 — rate limited; the only retryable status (with 529).
    RateLimit,
    /// HTTP 529 — the upstream is overloaded; the only other retryable status.
    Overloaded,
    /// Any other 5xx status.
    Server,
    /// Any status that does not map to a known class.
    Unexpected,
}

impl ApiErrorKind {
    /// Classify an HTTP status code.
    pub fn from_status(status: u16) -> Self {
        match status {
            401 => Self::Authentication,
            403 => Self::PermissionDenied,
            404 => Self::NotFound,
            422 => Self::RemoteValidation,
            429 => Self::RateLimit,
            529 => Self::Overloaded,
            500..=599 => Self::Server,
            _ => Self::Unexpected,
        }
    }

    /// A short, stable label.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Authentication => "authentication",
            Self::PermissionDenied => "permission denied",
            Self::NotFound => "not found",
            Self::RemoteValidation => "remote validation",
            Self::RateLimit => "rate limit",
            Self::Overloaded => "overloaded",
            Self::Server => "server",
            Self::Unexpected => "unexpected status",
        }
    }
}

impl std::fmt::Display for ApiErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Every error raised by `jev-client`.
///
/// The variants mirror the Julia client's conceptual hierarchy: configuration,
/// transport, protocol, API, retry-budget, and concurrency errors
/// (`docs/agents/specs/docs/17_JEV_CLIENT_DESIGN.md` §10). No variant stores a
/// credential, request body, or response body.
#[derive(Debug, Error)]
pub enum JevError {
    /// A configuration or constructor argument was rejected.
    #[error("configuration error: {message}")]
    Configuration {
        /// Human-readable, non-sensitive reason.
        message: String,
        /// Field path when one applies.
        field: Option<String>,
    },

    /// A credential provider failed or produced an invalid key.
    #[error("credential error: {message}")]
    Credential {
        /// Human-readable, non-sensitive reason.
        message: String,
    },

    /// The fixed endpoint policy was violated.
    #[error("endpoint policy error: {message}")]
    Endpoint {
        /// Human-readable, non-sensitive reason.
        message: String,
    },

    /// Local validation of the state, questions, or model failed.
    #[error("local validation error: {message}")]
    Validation {
        /// Human-readable, non-sensitive reason.
        message: String,
        /// Field path that failed.
        field: Option<String>,
    },

    /// The client has been closed.
    #[error("client is closed")]
    Closed,

    /// The transport failed in a non-TLS, non-timeout way.
    #[error("transport error: {message} (attempt {attempt})")]
    Transport {
        /// Human-readable, non-sensitive reason.
        message: String,
        /// One-based attempt count.
        attempt: u32,
    },

    /// TLS certificate or hostname verification failed.
    #[error("TLS verification failed (attempt {attempt})")]
    Tls {
        /// One-based attempt count.
        attempt: u32,
    },

    /// A connect, first-byte, or attempt timeout elapsed.
    #[error("timeout: {message} (attempt {attempt})")]
    Timeout {
        /// Human-readable, non-sensitive reason.
        message: String,
        /// One-based attempt count.
        attempt: u32,
    },

    /// The upstream returned a 3xx response; redirects are never followed.
    #[error("redirect refused (status {status})")]
    Redirect {
        /// The redirect status code.
        status: u16,
    },

    /// The response exceeded `max_response_bytes`.
    #[error("response too large")]
    ResponseTooLarge {
        /// Status when the rejection happened after headers were read.
        status: Option<u16>,
    },

    /// The response `Content-Type` was not JSON.
    #[error("unexpected content type")]
    UnexpectedContentType,

    /// The response body was not valid, bounded, duplicate-free JSON.
    #[error("malformed JSON: {message}")]
    MalformedJson {
        /// Human-readable, non-sensitive reason.
        message: String,
    },

    /// The response structurally parsed but failed cross-field validation.
    #[error("response validation error: {message}")]
    ResponseValidation {
        /// Human-readable, non-sensitive reason.
        message: String,
        /// Field path that failed.
        field: Option<String>,
    },

    /// The upstream returned a non-success status other than 429/529.
    #[error("API error ({kind}, status {status})")]
    Api {
        /// Upstream error class.
        kind: ApiErrorKind,
        /// HTTP status code.
        status: u16,
        /// Upstream request id when present.
        request_id: Option<String>,
        /// `Retry-After` value in seconds when present and valid.
        retry_after: Option<f64>,
        /// One-based attempt count.
        attempt: u32,
    },

    /// A retry could not proceed without exceeding the total retry budget.
    #[error("retry budget exceeded (attempt {attempt})")]
    RetryBudgetExceeded {
        /// One-based attempt count.
        attempt: u32,
    },

    /// A concurrency or lifecycle limit was violated.
    #[error("concurrency limit error: {message}")]
    Concurrency {
        /// Human-readable, non-sensitive reason.
        message: String,
    },
}

impl JevError {
    /// Build a configuration error.
    pub fn configuration(message: impl Into<String>) -> Self {
        Self::Configuration {
            message: message.into(),
            field: None,
        }
    }

    /// Build a configuration error with a field path.
    pub fn configuration_field(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Configuration {
            message: message.into(),
            field: Some(field.into()),
        }
    }

    /// Build a validation error.
    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation {
            message: message.into(),
            field: None,
        }
    }

    /// Build a validation error with a field path.
    pub fn validation_field(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Validation {
            message: message.into(),
            field: Some(field.into()),
        }
    }

    /// Build a response-validation error.
    pub fn response_validation(message: impl Into<String>) -> Self {
        Self::ResponseValidation {
            message: message.into(),
            field: None,
        }
    }

    /// Build a response-validation error with a field path.
    pub fn response_validation_field(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ResponseValidation {
            message: message.into(),
            field: Some(field.into()),
        }
    }
}

impl From<DecisionError> for JevError {
    fn from(error: DecisionError) -> Self {
        match error {
            DecisionError::InvalidInput { field, message } => Self::Validation { message, field },
            DecisionError::UnsupportedConfig { message } => Self::Configuration {
                message,
                field: None,
            },
            DecisionError::Backend { message, .. } => Self::Configuration {
                message,
                field: None,
            },
            DecisionError::Transport { message, .. } => Self::Transport {
                message,
                attempt: 0,
            },
            DecisionError::Remote { status, request_id } => Self::Api {
                kind: ApiErrorKind::from_status(status),
                status,
                request_id,
                retry_after: None,
                attempt: 0,
            },
        }
    }
}
