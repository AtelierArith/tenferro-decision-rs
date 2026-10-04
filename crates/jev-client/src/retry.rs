//! Retry helpers: the injectable sleeper and `Retry-After` parsing.
//!
//! Only HTTP 429 and 529 are retried, with exponential backoff and full jitter,
//! honoring `Retry-After`. TLS, redirect, JSON, validation, and ambiguous I/O
//! failures are never retried (`docs/agents/specs/docs/17_JEV_CLIENT_DESIGN.md`
//! §6, §10). The sleeper is injectable so tests are deterministic.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::errors::{JevError, Result};
use crate::transport::HttpResponse;

/// Maximum accepted `Retry-After` delay, in seconds.
pub const MAX_RETRY_AFTER_SECONDS: f64 = 86_400.0;

/// A blocking sleep primitive.
pub trait Sleeper: Send + Sync {
    /// Sleep for `duration`.
    fn sleep(&self, duration: Duration);
}

/// The default [`Sleeper`] backed by [`std::thread::sleep`].
#[derive(Debug, Default)]
pub struct RealSleeper;

impl Sleeper for RealSleeper {
    fn sleep(&self, duration: Duration) {
        std::thread::sleep(duration);
    }
}

/// A [`Sleeper`] that records requested durations without actually sleeping.
///
/// Inject it to keep retry tests deterministic and instant.
#[derive(Clone, Debug, Default)]
pub struct RecordingSleeper {
    sleeps: Arc<Mutex<Vec<Duration>>>,
}

impl RecordingSleeper {
    /// A fresh recorder.
    pub fn new() -> Self {
        Self::default()
    }

    /// The durations requested so far, in order.
    pub fn sleeps(&self) -> Vec<Duration> {
        self.sleeps.lock().expect("sleeper log poisoned").clone()
    }

    /// The number of sleep requests.
    pub fn count(&self) -> usize {
        self.sleeps.lock().expect("sleeper log poisoned").len()
    }
}

impl Sleeper for RecordingSleeper {
    fn sleep(&self, duration: Duration) {
        self.sleeps
            .lock()
            .expect("sleeper log poisoned")
            .push(duration);
    }
}

/// Read a `Retry-After` (seconds) or `retry-after-ms` header.
///
/// Returns `Ok(None)` when neither header is present. Invalid, negative, or
/// oversized values are rejected. HTTP-date `Retry-After` values are not
/// supported in 0.1 (a documented deviation) and are rejected rather than
/// guessed. `retry-after-ms` takes precedence when present.
pub fn parse_retry_after(response: &HttpResponse) -> Result<Option<Duration>> {
    if let Some(value) = response.header("retry-after-ms") {
        return parse_milliseconds(value).map(Some);
    }
    match response.header("retry-after") {
        Some(value) => parse_seconds(value).map(Some),
        None => Ok(None),
    }
}

fn parse_milliseconds(value: &str) -> Result<Duration> {
    let trimmed = value.trim();
    let millis: u64 = trimmed.parse().map_err(|_| invalid_retry_after())?;
    let duration = Duration::from_millis(millis);
    if duration > Duration::from_secs_f64(MAX_RETRY_AFTER_SECONDS) {
        return Err(invalid_retry_after());
    }
    Ok(duration)
}

fn parse_seconds(value: &str) -> Result<Duration> {
    let trimmed = value.trim();
    let seconds: f64 = trimmed.parse().map_err(|_| invalid_retry_after())?;
    if !seconds.is_finite() || seconds < 0.0 || seconds > MAX_RETRY_AFTER_SECONDS {
        return Err(invalid_retry_after());
    }
    Ok(Duration::from_secs_f64(seconds))
}

fn invalid_retry_after() -> JevError {
    JevError::ResponseValidation {
        message: "invalid Retry-After header".into(),
        field: Some("retry-after".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_seconds_and_milliseconds() {
        let response = HttpResponse::json(429, "").with_header("Retry-After", "2.5");
        assert_eq!(
            parse_retry_after(&response).unwrap(),
            Some(Duration::from_millis(2500))
        );

        let response = HttpResponse::json(429, "").with_header("retry-after-ms", "150");
        assert_eq!(
            parse_retry_after(&response).unwrap(),
            Some(Duration::from_millis(150))
        );

        assert_eq!(
            parse_retry_after(&HttpResponse::json(200, "")).unwrap(),
            None
        );
    }

    #[test]
    fn rejects_invalid_retry_after() {
        for value in ["-1", "NaN", "inf", "soon", "999999"] {
            let response = HttpResponse::json(429, "").with_header("Retry-After", value);
            assert!(parse_retry_after(&response).is_err(), "accepted {value}");
        }
    }
}
