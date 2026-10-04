use thiserror::Error;

/// Every error raised by `decision-core` and by engines implementing
/// [`DecisionEngine`](crate::DecisionEngine).
///
/// Error context never carries credentials, request bodies, or model inputs
/// (`docs/agents/specs/docs/07_SECURITY_LICENSE.md` §3). Engine- and
/// client-specific failures travel as an opaque [`Backend`](Self::Backend) or
/// [`Transport`](Self::Transport) source.
#[derive(Debug, Error)]
pub enum DecisionError {
    /// A question, state, or answer failed local validation.
    #[error("invalid input for `{}`: {message}", .field.as_deref().unwrap_or("<unspecified>"))]
    InvalidInput {
        /// Field path or name, at most the field that failed.
        field: Option<String>,
        /// Human-readable, non-sensitive reason.
        message: String,
    },

    /// The requested configuration is not supported by this build.
    #[error("unsupported configuration: {message}")]
    UnsupportedConfig {
        /// Human-readable reason.
        message: String,
    },

    /// A local execution backend failed.
    #[error("backend error: {message}")]
    Backend {
        /// Human-readable summary.
        message: String,
        /// Opaque underlying cause.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    },

    /// A transport-layer failure (network, TLS, timeout).
    #[error("transport error: {message}")]
    Transport {
        /// Human-readable summary.
        message: String,
        /// Opaque underlying cause.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    },

    /// The remote service rejected the request or returned an unusable result.
    #[error("remote error (status {status})")]
    Remote {
        /// HTTP status code when available.
        status: u16,
        /// Server-provided request identifier, when present.
        request_id: Option<String>,
    },
}

impl DecisionError {
    /// Build an invalid-input error without a field path.
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::InvalidInput {
            field: None,
            message: message.into(),
        }
    }

    /// Build an invalid-input error referring to a specific field.
    pub fn invalid_field(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::InvalidInput {
            field: Some(field.into()),
            message: message.into(),
        }
    }

    /// The field path, when one was recorded.
    pub fn field(&self) -> Option<&str> {
        match self {
            Self::InvalidInput { field, .. } => field.as_deref(),
            _ => None,
        }
    }
}

/// Convenience alias for fallible `decision-core` operations.
pub type Result<T, E = DecisionError> = std::result::Result<T, E>;
