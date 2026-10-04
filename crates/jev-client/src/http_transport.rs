//! The concrete HTTP/TLS transport backed by `reqwest` + `rustls`.
//!
//! Enabled by the non-default `http` cargo feature. This is the only part of
//! `jev-client` that touches the network, and it is deliberately narrow
//! (`docs/agents/specs/docs/17_JEV_CLIENT_DESIGN.md` §3–§4):
//!
//! - TLS certificate and hostname verification are always on; there is no
//!   public (or crate-internal) switch that turns them off.
//! - Redirects are never followed ([`Policy::none`]); a 3xx is surfaced as a
//!   normal response for [`crate::Client`] to reject.
//! - Proxies are off (`.no_proxy()`), so `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY`
//!   are ignored.
//! - reqwest's implicit protocol-NACK retry is disabled
//!   ([`reqwest::retry::never`]) and automatic decompression is disabled; all
//!   retry logic stays in [`crate::retry`].
//! - Response bodies are bounded: an oversized `Content-Length` is rejected
//!   before any body byte is read, and a streamed/chunked body is aborted as
//!   soon as the cap is exceeded. Compressed responses are rejected.
//!
//! The `Authorization` header is built here from the redacted
//! [`Secret`] so the credential never enters [`HttpRequest::headers`].

use std::collections::HashMap;
use std::io::Read;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use reqwest::Method;
use reqwest::blocking::{Client, ClientBuilder, Response};
use reqwest::header::{AUTHORIZATION, CONTENT_ENCODING, HeaderMap, HeaderName, HeaderValue};
use reqwest::redirect::Policy;

use crate::credentials::Secret;
use crate::transport::{
    HttpMethod, HttpRequest, HttpResponse, Transport, TransportError, endpoint_url,
};

/// The security-critical `reqwest` client configuration.
///
/// Isolated from the builder calls so the hardened choices are unit-testable
/// without constructing a client or touching the network. The type is
/// crate-private and [`ClientPolicy::hardened`] is the only constructor used in
/// production, so a caller cannot enable redirects, proxies, or certificate
/// bypass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ClientPolicy {
    /// Follow HTTP redirects (always `false` in production).
    pub(crate) follow_redirects: bool,
    /// Use the system/`*_PROXY` proxy (always `false` in production).
    pub(crate) use_proxy: bool,
}

impl ClientPolicy {
    /// The only policy the transport ever applies.
    pub(crate) const fn hardened() -> Self {
        Self {
            follow_redirects: false,
            use_proxy: false,
        }
    }
}

/// Apply a [`ClientPolicy`] to a reqwest builder.
///
/// Kept as a separate function so tests exercise exactly the configuration the
/// transport uses. Auto-decompression and the implicit protocol-NACK retry are
/// always disabled, independent of the policy fields.
fn configure(builder: ClientBuilder, policy: ClientPolicy) -> ClientBuilder {
    let builder = builder
        .redirect(if policy.follow_redirects {
            Policy::limited(10)
        } else {
            Policy::none()
        })
        .danger_accept_invalid_certs(false)
        .retry(reqwest::retry::never())
        .referer(false)
        .no_gzip()
        .no_brotli()
        .no_deflate()
        .no_zstd();
    if policy.use_proxy {
        builder
    } else {
        builder.no_proxy()
    }
}

/// The concrete HTTP/TLS [`Transport`] used by default with the `http` feature.
///
/// Holds one reqwest client per connect timeout so the connection pool is
/// reused across calls. The other timeouts travel per request: the whole
/// attempt is bounded by [`HttpRequest::attempt_timeout`]. Construction never
/// performs I/O.
#[derive(Debug, Default)]
pub struct ReqwestTransport {
    clients: Mutex<HashMap<Duration, Arc<Client>>>,
}

impl ReqwestTransport {
    /// Create a transport. No network activity occurs here.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return (building on first use) the client for `connect_timeout`.
    fn client(
        &self,
        connect_timeout: Duration,
    ) -> std::result::Result<Arc<Client>, TransportError> {
        let mut clients = self
            .clients
            .lock()
            .map_err(|_| TransportError::Io("transport state is unavailable".into()))?;
        if let Some(client) = clients.get(&connect_timeout) {
            return Ok(client.clone());
        }
        let client = configure(Client::builder(), ClientPolicy::hardened())
            .connect_timeout(connect_timeout)
            .build()
            .map_err(|_| TransportError::Connect("could not build the HTTP client".into()))?;
        let client = Arc::new(client);
        clients.insert(connect_timeout, client.clone());
        Ok(client)
    }
}

impl Transport for ReqwestTransport {
    fn execute(&self, request: &HttpRequest) -> std::result::Result<HttpResponse, TransportError> {
        // The only place a URL is formed; reject policy violations before any
        // network activity.
        let url = endpoint_url(&request.path)
            .map_err(|_| TransportError::Endpoint("unsupported endpoint path".into()))?;
        let client = self.client(request.connect_timeout)?;
        let method = match request.method {
            HttpMethod::Get => Method::GET,
            HttpMethod::Post => Method::POST,
        };

        let mut builder = client
            .request(method, url)
            .timeout(request.attempt_timeout)
            .headers(build_headers(request)?);
        if matches!(request.method, HttpMethod::Post) {
            builder = builder.body(request.body.clone());
        }

        let response = builder.send().map_err(|error| classify_error(&error))?;
        read_response(response, request.max_response_bytes)
    }
}

/// Build the request headers, adding `Authorization` from the redacted secret.
///
/// The fixed headers come from the client and are passed through unchanged;
/// this function only adds the bearer header, which the client deliberately
/// keeps out of [`HttpRequest::headers`].
fn build_headers(request: &HttpRequest) -> std::result::Result<HeaderMap, TransportError> {
    let mut headers = HeaderMap::new();
    for (name, value) in &request.headers {
        let name = HeaderName::from_bytes(name.as_bytes())
            .map_err(|_| TransportError::Io("invalid request header name".into()))?;
        let value = HeaderValue::from_str(value)
            .map_err(|_| TransportError::Io("invalid request header value".into()))?;
        headers.insert(name, value);
    }
    headers.insert(AUTHORIZATION, authorization_value(&request.credential)?);
    Ok(headers)
}

/// Render `Bearer <secret>` without leaving the key in a live [`String`].
fn authorization_value(secret: &Secret) -> std::result::Result<HeaderValue, TransportError> {
    let key = secret.expose_secret();
    let mut buffer = Vec::with_capacity(b"Bearer ".len() + key.len());
    buffer.extend_from_slice(b"Bearer ");
    buffer.extend_from_slice(key);
    let value = HeaderValue::from_bytes(&buffer);
    // Best-effort: overwrite the temporary copy after reqwest has taken its own,
    // on both the success and failure paths.
    buffer.fill(0);
    value.map_err(|_| TransportError::Io("invalid credential".into()))
}

/// Read a response into the crate's bounded [`HttpResponse`].
///
/// The transport never buffers more than `max_response_bytes + 1` bytes. If the
/// declared `Content-Length` exceeds the cap the body is not read at all, and a
/// streamed body is aborted once the cap is crossed; in both cases the bounded
/// response is handed back so [`crate::Client`] can raise its typed
/// `ResponseTooLarge` error. Compressed responses are rejected outright.
fn read_response(
    mut response: Response,
    max_response_bytes: usize,
) -> std::result::Result<HttpResponse, TransportError> {
    let status = response.status().as_u16();
    let headers = response_headers(response.headers());

    if let Some(encoding) = response.headers().get(CONTENT_ENCODING) {
        if !is_identity_encoding(encoding) {
            return Err(TransportError::Io(
                "compressed responses are rejected".into(),
            ));
        }
    }

    // Reject an oversized declared length before touching the body.
    if response
        .content_length()
        .is_some_and(|length| length > max_response_bytes as u64)
    {
        return Ok(HttpResponse::new(status, headers, Vec::new()));
    }

    let body = read_capped(&mut response, max_response_bytes)
        .map_err(|_| TransportError::Io("failed reading the response body".into()))?;
    Ok(HttpResponse::new(status, headers, body))
}

fn response_headers(headers: &HeaderMap) -> Vec<(String, String)> {
    headers
        .iter()
        .map(|(name, value)| {
            (
                name.as_str().to_string(),
                value.to_str().unwrap_or_default().to_string(),
            )
        })
        .collect()
}

/// `true` when every comma-separated token is empty or `identity`.
fn is_identity_encoding(value: &HeaderValue) -> bool {
    let Ok(value) = value.to_str() else {
        return false;
    };
    value.split(',').all(|part| {
        let part = part.trim().to_ascii_lowercase();
        part.is_empty() || part == "identity"
    })
}

/// Read at most `max_response_bytes + 1` bytes from `reader`.
///
/// Stopping one byte past the cap both bounds memory and lets the caller detect
/// that the body overflowed without consuming the rest of the stream.
fn read_capped<R: Read>(reader: &mut R, max_response_bytes: usize) -> std::io::Result<Vec<u8>> {
    let limit = max_response_bytes.saturating_add(1);
    let mut body = Vec::with_capacity(limit.min(64 * 1024));
    let mut chunk = [0u8; 8192];
    while body.len() < limit {
        let remaining = limit - body.len();
        let want = remaining.min(chunk.len());
        let read = reader.read(&mut chunk[..want])?;
        if read == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..read]);
    }
    Ok(body)
}

/// Map a reqwest error onto the crate's [`TransportError`] variants.
///
/// Only static, non-sensitive messages are used, so a request body or URL is
/// never copied into an error.
fn classify_error(error: &reqwest::Error) -> TransportError {
    if error.is_timeout() {
        return TransportError::Timeout("the request timed out".into());
    }
    if error.is_connect() {
        if error_chain_looks_like_tls(error) {
            return TransportError::Tls("TLS verification failed".into());
        }
        return TransportError::Connect("could not connect to the API endpoint".into());
    }
    if error.is_redirect() {
        return TransportError::Io("unexpected redirect".into());
    }
    if error.is_body() || error.is_decode() {
        return TransportError::Io("failed reading the response body".into());
    }
    TransportError::Io("transport failure".into())
}

/// Walk the error source chain looking for TLS/certificate wording.
///
/// reqwest does not expose a typed TLS error, so this is a best-effort
/// classification. Correctness does not depend on it: TLS failures are never
/// retried because the client only retries HTTP 429/529.
fn error_chain_looks_like_tls(error: &reqwest::Error) -> bool {
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(error) = current {
        if message_looks_like_tls(&error.to_string()) {
            return true;
        }
        current = error.source();
    }
    false
}

fn message_looks_like_tls(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    [
        "certificate",
        "tls",
        "handshake",
        "unknownissuer",
        "invalid peer",
    ]
    .iter()
    .any(|needle| message.contains(needle))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample_request(path: &str) -> HttpRequest {
        HttpRequest {
            method: HttpMethod::Post,
            path: path.to_string(),
            headers: vec![
                (
                    "Content-Type".into(),
                    "application/json; charset=utf-8".into(),
                ),
                ("Accept-Encoding".into(), "identity".into()),
            ],
            credential: Secret::new(b"test-key-abc-123".to_vec()),
            body: b"{\"state\":\"x\"}".to_vec(),
            connect_timeout: Duration::from_secs(1),
            first_byte_timeout: Duration::from_secs(1),
            attempt_timeout: Duration::from_secs(1),
            max_response_bytes: 1024,
        }
    }

    #[test]
    fn hardened_policy_disables_redirects_and_proxies() {
        let policy = ClientPolicy::hardened();
        assert!(!policy.follow_redirects);
        assert!(!policy.use_proxy);
    }

    #[test]
    fn configured_builder_is_constructible() {
        // Building the builder performs no I/O and no TLS handshake.
        let _builder = configure(Client::builder(), ClientPolicy::hardened());
    }

    #[test]
    fn rejects_paths_outside_the_fixed_endpoint() {
        let transport = ReqwestTransport::new();
        for path in [
            "https://attacker.example/v1/systemone",
            "/v1/models/../../evil",
            "",
        ] {
            let error = transport.execute(&sample_request(path)).unwrap_err();
            assert!(
                matches!(error, TransportError::Endpoint(_)),
                "expected endpoint rejection for {path:?}, got {error:?}"
            );
        }
    }

    #[test]
    fn authorization_header_is_built_and_secret_is_never_formatted() {
        let secret = Secret::new(b"sentinel-secret-value".to_vec());
        let value = authorization_value(&secret).unwrap();
        assert_eq!(value.to_str().unwrap(), "Bearer sentinel-secret-value");
        assert_eq!(format!("{secret}"), "<redacted>");
        assert_eq!(format!("{secret:?}"), "<redacted>");
    }

    #[test]
    fn invalid_url_maps_to_a_typed_transport_error() {
        // `https://` fails to parse locally, so this never opens a socket.
        let error = Client::new()
            .get("https://")
            .send()
            .expect_err("an empty host must fail before connecting");
        let mapped = classify_error(&error);
        assert!(matches!(mapped, TransportError::Io(_)));
        assert!(!format!("{mapped}").contains("sentinel-secret-value"));
        assert!(!format!("{mapped}").contains("body"));
    }

    #[test]
    fn tls_wording_is_detected_in_an_error_chain() {
        assert!(message_looks_like_tls(
            "invalid peer certificate: UnknownIssuer"
        ));
        assert!(message_looks_like_tls("TLS handshake failure"));
        assert!(!message_looks_like_tls("connection refused"));
        assert!(!message_looks_like_tls(
            "error sending request for url (https://api.typesafe.ai:443/v1/systemone)"
        ));
    }

    #[test]
    fn only_identity_content_encoding_is_accepted() {
        for value in ["identity", "identity, identity", "identity , Identity"] {
            assert!(
                is_identity_encoding(&HeaderValue::from_str(value).unwrap()),
                "expected {value:?} to be accepted"
            );
        }
        for value in ["gzip", "br", "gzip, identity", "deflate"] {
            assert!(
                !is_identity_encoding(&HeaderValue::from_str(value).unwrap()),
                "expected {value:?} to be rejected"
            );
        }
    }

    #[test]
    fn read_capped_stops_one_byte_past_the_cap() {
        let data = vec![b'x'; 1000];
        let body = read_capped(&mut Cursor::new(data), 16).unwrap();
        assert_eq!(body.len(), 17);
        assert!(body.iter().all(|byte| *byte == b'x'));
    }

    #[test]
    fn read_capped_returns_short_bodies_intact() {
        let body = read_capped(&mut Cursor::new(b"hello".to_vec()), 16).unwrap();
        assert_eq!(body, b"hello");
    }
}
