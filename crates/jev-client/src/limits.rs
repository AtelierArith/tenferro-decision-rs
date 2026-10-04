//! Resource, timeout, and retry policies.
//!
//! Defaults and constraints follow `docs/agents/specs/docs/17_JEV_CLIENT_DESIGN.md`
//! §6. Limits are enforced locally and are never relaxed per call; build a new
//! [`Client`](crate::Client) to change them.

use std::time::Duration;

use crate::errors::{JevError, Result};

/// Bounded-resource limits for one client.
///
/// All values must be positive. Defaults follow the design document:
/// 1 MiB request, 8 MiB response, 64 KiB error body, JSON depth 32, 1 MiB
/// string, 100 000 container items, 1024 questions, 128-character question id,
/// 128-byte model id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceLimits {
    /// Maximum serialized request body size in bytes.
    pub max_request_bytes: usize,
    /// Maximum response body size in bytes.
    pub max_response_bytes: usize,
    /// Maximum error-body size in bytes (used by the concrete transport).
    pub max_error_body_bytes: usize,
    /// Maximum JSON nesting depth for requests and responses.
    pub max_json_depth: usize,
    /// Maximum UTF-8 byte length of any single string.
    pub max_string_bytes: usize,
    /// Maximum number of items in any array or object.
    pub max_container_items: usize,
    /// Maximum number of questions per request.
    pub max_questions: usize,
    /// Maximum number of Unicode scalar values in a question id.
    pub max_question_id_chars: usize,
    /// Maximum model id length in bytes.
    pub max_model_id_bytes: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_request_bytes: 1024 * 1024,
            max_response_bytes: 8 * 1024 * 1024,
            max_error_body_bytes: 64 * 1024,
            max_json_depth: 32,
            max_string_bytes: 1024 * 1024,
            max_container_items: 100_000,
            max_questions: 1024,
            max_question_id_chars: 128,
            max_model_id_bytes: 128,
        }
    }
}

impl ResourceLimits {
    /// Validate that every limit is positive.
    pub fn validate(&self) -> Result<()> {
        let fields = [
            ("max_request_bytes", self.max_request_bytes),
            ("max_response_bytes", self.max_response_bytes),
            ("max_error_body_bytes", self.max_error_body_bytes),
            ("max_json_depth", self.max_json_depth),
            ("max_string_bytes", self.max_string_bytes),
            ("max_container_items", self.max_container_items),
            ("max_questions", self.max_questions),
            ("max_question_id_chars", self.max_question_id_chars),
            ("max_model_id_bytes", self.max_model_id_bytes),
        ];
        for (name, value) in fields {
            if value == 0 {
                return Err(JevError::configuration_field(
                    name,
                    "resource limits must be positive",
                ));
            }
        }
        Ok(())
    }
}

/// Per-call timeouts in seconds.
///
/// Must satisfy `connect <= attempt <= total` and `first_byte <= attempt`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeoutPolicy {
    /// TCP/TLS connection timeout.
    pub connect: Duration,
    /// Time allowed for the first response byte.
    pub first_byte: Duration,
    /// Whole-attempt timeout.
    pub attempt: Duration,
    /// Whole-call timeout, including semaphore wait and retries.
    pub total: Duration,
}

impl Default for TimeoutPolicy {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(5),
            first_byte: Duration::from_secs(15),
            attempt: Duration::from_secs(30),
            total: Duration::from_secs(60),
        }
    }
}

impl TimeoutPolicy {
    /// Validate positivity and ordering.
    pub fn validate(&self) -> Result<()> {
        if self.connect.is_zero()
            || self.first_byte.is_zero()
            || self.attempt.is_zero()
            || self.total.is_zero()
        {
            return Err(JevError::configuration_field(
                "timeout",
                "timeout values must be positive",
            ));
        }
        if self.connect > self.attempt || self.attempt > self.total {
            return Err(JevError::configuration_field(
                "timeout",
                "timeout values must satisfy connect <= attempt <= total",
            ));
        }
        if self.first_byte > self.attempt {
            return Err(JevError::configuration_field(
                "timeout",
                "first_byte must not exceed attempt",
            ));
        }
        Ok(())
    }
}

/// Retry settings for the fixed 429/529 retry policy.
///
/// `max_retries` must be between 0 and 5. Delays and the total budget follow
/// `initial_delay <= max_delay <= total_budget <= 300 s`. The retryable status
/// set and ambiguous-I/O behavior are not configurable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Additional attempts permitted after the first send (0..=5).
    pub max_retries: u32,
    /// Base exponential-backoff delay.
    pub initial_delay: Duration,
    /// Maximum single backoff delay.
    pub max_delay: Duration,
    /// Total planned backoff budget.
    pub total_budget: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 2,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(15),
            total_budget: Duration::from_secs(60),
        }
    }
}

impl RetryPolicy {
    /// Validate `max_retries` and the delay/budget ordering.
    pub fn validate(&self) -> Result<()> {
        if self.max_retries > 5 {
            return Err(JevError::configuration_field(
                "retry.max_retries",
                "max_retries must be between 0 and 5",
            ));
        }
        if self.initial_delay > self.max_delay
            || self.max_delay > self.total_budget
            || self.total_budget > Duration::from_secs(300)
        {
            return Err(JevError::configuration_field(
                "retry",
                "retry delays and budget must satisfy initial_delay <= max_delay <= total_budget <= 300s",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_validate() {
        ResourceLimits::default().validate().unwrap();
        TimeoutPolicy::default().validate().unwrap();
        RetryPolicy::default().validate().unwrap();
    }

    #[test]
    fn rejects_zero_limits() {
        let limits = ResourceLimits {
            max_json_depth: 0,
            ..ResourceLimits::default()
        };
        assert!(limits.validate().is_err());
    }

    #[test]
    fn rejects_bad_timeout_ordering() {
        let policy = TimeoutPolicy {
            connect: Duration::from_secs(40),
            ..TimeoutPolicy::default()
        };
        assert!(policy.validate().is_err());
    }

    #[test]
    fn rejects_retry_out_of_range() {
        let policy = RetryPolicy {
            max_retries: 6,
            ..RetryPolicy::default()
        };
        assert!(policy.validate().is_err());
    }
}
