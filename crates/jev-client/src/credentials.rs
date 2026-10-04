//! Credential providers, validation, and redaction.
//!
//! The API key never appears in `Debug`, `Display`, errors, or logs. Providers
//! render as `<redacted>` and [`Secret`] masks its bytes. Validation mirrors
//! `docs/agents/specs/docs/17_JEV_CLIENT_DESIGN.md` §5: trim ASCII whitespace,
//! then reject empty, internal whitespace, control characters, non-ASCII, and
//! values longer than 4 KiB.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use crate::errors::{JevError, Result};

/// Maximum accepted API key length in bytes, after trimming.
pub const MAX_CREDENTIAL_BYTES: usize = 4096;

/// Default environment variable read by [`EnvCredential`].
pub const DEFAULT_ENV_VAR: &str = "TYPESAFE_API_KEY";

const REDACTED: &str = "<redacted>";

fn is_ascii_whitespace(byte: u8) -> bool {
    matches!(byte, 0x09 | 0x0a | 0x0b | 0x0c | 0x0d | 0x20)
}

/// Validate and normalize an API key, returning a redacted [`Secret`].
///
/// The key is trimmed of leading/trailing ASCII whitespace. The trimmed value
/// must be non-empty, at most [`MAX_CREDENTIAL_BYTES`], and contain only
/// printable ASCII (`0x21..=0x7e`). No prefix or minimum length is assumed.
pub fn validate_credential(value: &str) -> Result<Secret> {
    let raw = value.as_bytes();
    let mut left = 0;
    let mut right = raw.len();
    while left < right && is_ascii_whitespace(raw[left]) {
        left += 1;
    }
    while right > left && is_ascii_whitespace(raw[right - 1]) {
        right -= 1;
    }
    if left >= right {
        return Err(JevError::Credential {
            message: "API key is missing or empty".into(),
        });
    }
    let trimmed = &raw[left..right];
    if trimmed.len() > MAX_CREDENTIAL_BYTES {
        return Err(JevError::Credential {
            message: "API key is invalid".into(),
        });
    }
    if !trimmed.iter().all(|byte| (0x21..=0x7e).contains(byte)) {
        return Err(JevError::Credential {
            message: "API key is invalid".into(),
        });
    }
    Ok(Secret::new(trimmed.to_vec()))
}

/// A redacted byte container for an API key.
///
/// Bytes are only reachable through [`Secret::expose_secret`], which exists for
/// the narrow purpose of building the `Authorization` header. `Debug` and
/// `Display` always render `<redacted>`. Dropping the value overwrites its
/// buffer on a best-effort basis; this is a defensive measure, not a
/// cryptographic guarantee (`docs/agents/specs/docs/07_SECURITY_LICENSE.md`
/// §3).
#[derive(Clone)]
pub struct Secret(Vec<u8>);

impl Secret {
    /// Wrap already-validated key bytes.
    pub fn new(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    /// Expose the key bytes. Callers must not log or persist the result.
    pub fn expose_secret(&self) -> &[u8] {
        &self.0
    }

    /// `true` when the secret has no bytes.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Consume the secret and return its bytes, leaving the dropped buffer
    /// empty.
    pub(crate) fn into_bytes(mut self) -> Vec<u8> {
        std::mem::take(&mut self.0)
    }
}

impl Drop for Secret {
    fn drop(&mut self) {
        // Best-effort zeroization without `unsafe`; the optimizer may elide this.
        self.0.fill(0);
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(REDACTED)
    }
}

impl fmt::Display for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(REDACTED)
    }
}

/// A source of API keys invoked immediately before each request.
///
/// Implementations must be safe to share across threads. `close` stops further
/// use; providers used by a [`Client`](crate::Client) are closed when the client
/// is closed.
pub trait CredentialProvider: Send + Sync {
    /// Produce a validated secret, or a redacted [`JevError::Credential`].
    fn credential(&self) -> Result<Secret>;

    /// Stop the provider. Idempotent.
    fn close(&self) {}
}

/// Reads the API key from an environment variable at request time.
///
/// The variable name must be non-empty ASCII without control characters. The
/// value is read and validated on every call, so key rotation is picked up
/// automatically. Missing and empty values are the same error.
pub struct EnvCredential {
    name: String,
    closed: AtomicBool,
}

impl EnvCredential {
    /// Build a provider for `name`.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            closed: AtomicBool::new(false),
        }
    }

    /// Build a provider using [`DEFAULT_ENV_VAR`] but validate the name.
    pub fn validate(&self) -> Result<()> {
        let name = self.name.as_str();
        if name.is_empty()
            || !name.is_ascii()
            || name.bytes().any(|byte| byte < 0x20 || byte == 0x7f)
        {
            return Err(JevError::Credential {
                message: "environment variable name is invalid".into(),
            });
        }
        Ok(())
    }
}

impl fmt::Debug for EnvCredential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(REDACTED)
    }
}

impl CredentialProvider for EnvCredential {
    fn credential(&self) -> Result<Secret> {
        self.validate()?;
        if self.closed.load(Ordering::SeqCst) {
            return Err(JevError::Credential {
                message: "credential provider is closed".into(),
            });
        }
        let value = std::env::var(&self.name).map_err(|_| JevError::Credential {
            message: "API key is missing or empty".into(),
        })?;
        validate_credential(&value)
    }

    fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }
}

/// Owns an in-memory copy of an API key.
///
/// `close` overwrites and clears the stored bytes on a best-effort basis.
/// Prefer [`EnvCredential`] or [`CredentialCallback`] so the key is not held in
/// process memory longer than necessary.
pub struct StaticCredential {
    bytes: Mutex<Vec<u8>>,
    closed: AtomicBool,
}

impl StaticCredential {
    /// Validate and store a static key.
    pub fn new(secret: &str) -> Result<Self> {
        let secret = validate_credential(secret)?;
        Ok(Self {
            bytes: Mutex::new(secret.into_bytes()),
            closed: AtomicBool::new(false),
        })
    }
}

impl fmt::Debug for StaticCredential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(REDACTED)
    }
}

impl CredentialProvider for StaticCredential {
    fn credential(&self) -> Result<Secret> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(JevError::Credential {
                message: "credential provider is closed".into(),
            });
        }
        let guard = self.bytes.lock().map_err(|_| JevError::Credential {
            message: "credential provider is unavailable".into(),
        })?;
        if guard.is_empty() {
            return Err(JevError::Credential {
                message: "credential provider is closed".into(),
            });
        }
        Ok(Secret::new(guard.clone()))
    }

    fn close(&self) {
        if let Ok(mut guard) = self.bytes.lock() {
            guard.fill(0);
            guard.clear();
        }
        self.closed.store(true, Ordering::SeqCst);
    }
}

/// Error type accepted from a [`CredentialCallback`].
pub type CredentialCallbackError = Box<dyn std::error::Error + Send + Sync + 'static>;

type Callback = dyn Fn() -> std::result::Result<String, CredentialCallbackError> + Send + Sync;

/// Calls a user function before each request.
///
/// The callback returns the API key as a `String`; any error becomes a redacted
/// [`JevError::Credential`] and is not retained. Use this to integrate an OS
/// keychain or key-management service. The external store's lifetime is owned by
/// the caller and is not closed by the client.
pub struct CredentialCallback {
    callback: Box<Callback>,
    closed: AtomicBool,
}

impl CredentialCallback {
    /// Build a callback provider.
    pub fn new<F, E>(callback: F) -> Self
    where
        F: Fn() -> std::result::Result<String, E> + Send + Sync + 'static,
        E: std::error::Error + Send + Sync + 'static,
    {
        Self {
            callback: Box::new(move || callback().map_err(|error| Box::new(error) as _)),
            closed: AtomicBool::new(false),
        }
    }
}

impl fmt::Debug for CredentialCallback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(REDACTED)
    }
}

impl CredentialProvider for CredentialCallback {
    fn credential(&self) -> Result<Secret> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(JevError::Credential {
                message: "credential provider is closed".into(),
            });
        }
        let value = (self.callback)().map_err(|_| JevError::Credential {
            message: "credential callback failed".into(),
        })?;
        validate_credential(&value)
    }

    fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trims_ascii_whitespace() {
        let secret = validate_credential("  \t key-123 \n").unwrap();
        assert_eq!(secret.expose_secret(), b"key-123");
    }

    #[test]
    fn rejects_invalid_keys() {
        assert!(validate_credential("").is_err());
        assert!(validate_credential("   ").is_err());
        assert!(validate_credential("a b").is_err());
        assert!(validate_credential("a\tb").is_err());
        assert!(validate_credential("caf\u{e9}").is_err());
        assert!(validate_credential(&"x".repeat(MAX_CREDENTIAL_BYTES + 1)).is_err());
        assert!(validate_credential(&"x".repeat(MAX_CREDENTIAL_BYTES)).is_ok());
    }

    #[test]
    fn secret_is_redacted() {
        let secret = Secret::new(b"sentinel-secret-value".to_vec());
        assert_eq!(format!("{secret}"), REDACTED);
        assert_eq!(format!("{secret:?}"), REDACTED);
    }

    #[test]
    fn static_credential_zeroizes_on_close() {
        let provider = StaticCredential::new("sentinel-secret-value").unwrap();
        assert_eq!(
            provider.credential().unwrap().expose_secret(),
            b"sentinel-secret-value"
        );
        provider.close();
        assert!(provider.credential().is_err());
        assert!(format!("{provider:?}").contains(REDACTED));
    }

    #[test]
    fn callback_error_is_redacted() {
        let provider = CredentialCallback::new(|| -> std::result::Result<String, JevError> {
            Err(JevError::validation("boom"))
        });
        let error = provider.credential().unwrap_err();
        assert!(!format!("{error}").contains("boom"));
    }
}
