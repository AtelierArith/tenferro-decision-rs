//! The [`Client`] and [`ClientBuilder`]: request execution, lifecycle, and
//! concurrency.
//!
//! `Client` is shareable across threads. One in-flight slot is reserved per
//! call from a counting semaphore (`max_inflight`, default 8). [`Client::close`]
//! stops new calls and waits for in-flight calls without cancelling them; later
//! calls return [`JevError::Closed`]
//! (`docs/agents/specs/docs/17_JEV_CLIENT_DESIGN.md` §10, §17).

use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use decision_core::{QuestionSet, State, SystemOneResponse};

use crate::credentials::{CredentialProvider, EnvCredential, DEFAULT_ENV_VAR};
use crate::errors::{ApiErrorKind, JevError, Result};
use crate::limits::{ResourceLimits, RetryPolicy, TimeoutPolicy};
use crate::logging::{Event, EventSink, NoopEventSink, Operation};
use crate::models::{ModelList, ModelRef};
use crate::responses::{parse_model_list, parse_system_one_response};
use crate::retry::{parse_retry_after, RealSleeper, Sleeper};
use crate::serialization::serialize_request;
#[cfg(not(feature = "http"))]
use crate::transport::UnsupportedTransport;
use crate::transport::{
    endpoint_url, HttpMethod, HttpRequest, HttpResponse, Transport, TransportError, MODELS_PATH,
    SYSTEM_ONE_PATH, USER_AGENT,
};

const DEFAULT_MAX_INFLIGHT: usize = 8;
const DEFAULT_RNG_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

/// The transport used when the caller does not inject one.
///
/// The concrete reqwest/rustls transport is the default only with the `http`
/// feature; without it, a stub that always errors keeps default builds
/// dependency-light. An explicit [`ClientBuilder::transport`] always wins.
#[cfg(feature = "http")]
fn default_transport() -> Arc<dyn Transport> {
    Arc::new(crate::http_transport::ReqwestTransport::new())
}

#[cfg(not(feature = "http"))]
fn default_transport() -> Arc<dyn Transport> {
    Arc::new(UnsupportedTransport)
}

/// A System One request: the shared state plus the questions to answer.
///
/// The model is owned by the [`Client`], never by the request.
#[derive(Clone, Copy, Debug)]
pub struct SystemOneRequest<'a> {
    /// The state to evaluate against.
    pub state: &'a State,
    /// The questions to answer, in order.
    pub questions: &'a QuestionSet,
}

impl<'a> SystemOneRequest<'a> {
    /// Build a request from a state and a question set.
    pub fn new(state: &'a State, questions: &'a QuestionSet) -> Self {
        Self { state, questions }
    }
}

/// A shareable client for the TypeSafe AI System One API.
pub struct Client {
    model: ModelRef,
    credential: Arc<dyn CredentialProvider>,
    retry: RetryPolicy,
    timeout: TimeoutPolicy,
    limits: ResourceLimits,
    transport: Arc<dyn Transport>,
    sleeper: Arc<dyn Sleeper>,
    events: Arc<dyn EventSink>,
    lifecycle: Lifecycle,
    rng: Mutex<u64>,
}

impl Client {
    /// Start building a client. A model must be set before [`ClientBuilder::build`].
    pub fn builder() -> ClientBuilder {
        ClientBuilder::default()
    }

    /// The model reference this client sends.
    pub fn model(&self) -> &ModelRef {
        &self.model
    }

    /// `true` until [`Client::close`] is called.
    pub fn is_open(&self) -> bool {
        self.lifecycle.is_open()
    }

    /// Stop accepting new calls and wait for in-flight calls to finish.
    ///
    /// In-flight calls are not cancelled. Idempotent. The owned credential
    /// provider is closed, zeroizing a [`crate::StaticCredential`] on a
    /// best-effort basis.
    pub fn close(&self) {
        self.lifecycle.close();
        self.credential.close();
    }

    /// Evaluate a System One request.
    pub fn system_one(&self, request: &SystemOneRequest<'_>) -> Result<SystemOneResponse> {
        let started = Instant::now();
        let _slot = self.lifecycle.acquire()?;
        let outcome = self.perform(
            HttpMethod::Post,
            SYSTEM_ONE_PATH,
            || serialize_request(request.state, &self.model, request.questions, &self.limits),
            &self.retry,
            &self.timeout,
        )?;
        check_json_content_type(&outcome.response)?;
        let request_id = outcome.response.header("x-request-id").map(str::to_string);
        let parsed = parse_system_one_response(
            &outcome.response.body,
            request.questions,
            &self.limits,
            request_id,
        )?;
        self.emit(Event {
            operation: Operation::SystemOne,
            requested_model: self.model.as_str(),
            returned_model: Some(parsed.model.as_str()),
            moving_alias: self.model.is_moving_alias(),
            question_count: request.questions.len(),
            request_bytes: outcome.request_bytes,
            response_bytes: outcome.response.body.len(),
            status: Some(outcome.response.status),
            attempt: outcome.attempt,
            retried_status: outcome.retried_status,
            request_id: parsed.request_id.as_deref(),
            usage: Some(parsed.usage),
            latency: Some(started.elapsed()),
        });
        Ok(parsed)
    }

    /// List the models available to this client.
    pub fn list_models(&self) -> Result<ModelList> {
        let started = Instant::now();
        let _slot = self.lifecycle.acquire()?;
        let outcome = self.perform(
            HttpMethod::Get,
            MODELS_PATH,
            || Ok(Vec::new()),
            &self.retry,
            &self.timeout,
        )?;
        check_json_content_type(&outcome.response)?;
        let models = parse_model_list(&outcome.response.body, &self.limits)?;
        let request_id = outcome.response.header("x-request-id").map(str::to_string);
        self.emit(Event {
            operation: Operation::ListModels,
            requested_model: self.model.as_str(),
            returned_model: None,
            moving_alias: self.model.is_moving_alias(),
            question_count: 0,
            request_bytes: outcome.request_bytes,
            response_bytes: outcome.response.body.len(),
            status: Some(outcome.response.status),
            attempt: outcome.attempt,
            retried_status: outcome.retried_status,
            request_id: request_id.as_deref(),
            usage: None,
            latency: Some(started.elapsed()),
        });
        Ok(models)
    }

    /// Run one attempt loop, rebuilding the body and credential each attempt.
    fn perform<F>(
        &self,
        method: HttpMethod,
        path: &str,
        make_body: F,
        retry: &RetryPolicy,
        timeout: &TimeoutPolicy,
    ) -> Result<AttemptOutcome>
    where
        F: Fn() -> Result<Vec<u8>>,
    {
        let mut cumulative = Duration::ZERO;
        let mut attempt = 0u32;
        let mut retried_status = None;
        loop {
            attempt += 1;
            let body = make_body()?;
            let request_bytes = body.len();
            let credential = self.credential.credential()?;
            let request = HttpRequest {
                method,
                path: path.to_string(),
                headers: fixed_headers(),
                credential,
                body,
                connect_timeout: timeout.connect,
                first_byte_timeout: timeout.first_byte,
                attempt_timeout: timeout.attempt,
                max_response_bytes: self.limits.max_response_bytes,
            };
            let response = self
                .transport
                .execute(&request)
                .map_err(|error| map_transport_error(error, attempt))?;
            self.check_response_size(&response)?;
            let retry_after = parse_retry_after(&response)?;
            let status = response.status;

            if (300..=399).contains(&status) {
                return Err(JevError::Redirect { status });
            }
            if status == 200 {
                return Ok(AttemptOutcome {
                    response,
                    attempt,
                    request_bytes,
                    retried_status,
                });
            }
            if (status == 429 || status == 529) && attempt <= retry.max_retries {
                let delay = self.retry_delay(retry, attempt, retry_after);
                let remaining = retry.total_budget.saturating_sub(cumulative);
                if delay > remaining {
                    return Err(JevError::RetryBudgetExceeded { attempt });
                }
                cumulative += delay;
                retried_status = Some(status);
                self.sleeper.sleep(delay);
                continue;
            }
            let request_id = response.header("x-request-id").map(str::to_string);
            return Err(api_error(status, request_id, retry_after, attempt));
        }
    }

    fn check_response_size(&self, response: &HttpResponse) -> Result<()> {
        if let Some(value) = response.header("content-length") {
            if let Ok(length) = value.trim().parse::<u64>() {
                if length > self.limits.max_response_bytes as u64 {
                    return Err(JevError::ResponseTooLarge {
                        status: Some(response.status),
                    });
                }
            }
        }
        if response.body.len() > self.limits.max_response_bytes {
            return Err(JevError::ResponseTooLarge {
                status: Some(response.status),
            });
        }
        Ok(())
    }

    fn retry_delay(
        &self,
        retry: &RetryPolicy,
        attempt: u32,
        retry_after: Option<Duration>,
    ) -> Duration {
        let exponent = attempt.saturating_sub(1).min(16);
        let base = retry
            .initial_delay
            .saturating_mul(1u32 << exponent)
            .min(retry.max_delay);
        let jittered = base.mul_f64(self.next_jitter());
        let server = retry_after.unwrap_or(Duration::ZERO);
        jittered.max(server)
    }

    /// A deterministic, seedable jitter source (xorshift), in `[0, 1)`.
    fn next_jitter(&self) -> f64 {
        let mut state = self.rng.lock().expect("jitter state poisoned");
        let mut value = *state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        *state = value;
        (value >> 11) as f64 / (1u64 << 53) as f64
    }

    fn emit(&self, event: Event<'_>) {
        self.events.on_event(&event);
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        self.close();
    }
}

struct AttemptOutcome {
    response: HttpResponse,
    attempt: u32,
    request_bytes: usize,
    retried_status: Option<u16>,
}

fn fixed_headers() -> Vec<(String, String)> {
    vec![
        (
            "Content-Type".into(),
            "application/json; charset=utf-8".into(),
        ),
        ("Accept".into(), "application/json".into()),
        ("Accept-Encoding".into(), "identity".into()),
        ("User-Agent".into(), USER_AGENT.into()),
    ]
}

fn check_json_content_type(response: &HttpResponse) -> Result<()> {
    let content_type = response
        .header("content-type")
        .ok_or(JevError::UnexpectedContentType)?;
    let essence = content_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if essence == "application/json"
        || (essence.starts_with("application/") && essence.ends_with("+json"))
    {
        Ok(())
    } else {
        Err(JevError::UnexpectedContentType)
    }
}

fn map_transport_error(error: TransportError, attempt: u32) -> JevError {
    match error {
        TransportError::Tls(_) => JevError::Tls { attempt },
        TransportError::Timeout(message) => JevError::Timeout { message, attempt },
        TransportError::Endpoint(message) => JevError::Endpoint { message },
        other => JevError::Transport {
            message: other.to_string(),
            attempt,
        },
    }
}

fn api_error(
    status: u16,
    request_id: Option<String>,
    retry_after: Option<Duration>,
    attempt: u32,
) -> JevError {
    JevError::Api {
        kind: ApiErrorKind::from_status(status),
        status,
        request_id,
        retry_after: retry_after.map(|delay| delay.as_secs_f64()),
        attempt,
    }
}

/// Build a client, run `f`, then always close the client.
///
/// The client is closed even if `f` panics, via [`Client`]'s `Drop`.
pub fn with_client<F, T>(builder: ClientBuilder, f: F) -> Result<T>
where
    F: FnOnce(&Client) -> T,
{
    let client = builder.build()?;
    let output = f(&client);
    client.close();
    Ok(output)
}

/// Builder for [`Client`].
pub struct ClientBuilder {
    model: Option<ModelRef>,
    credential: Arc<dyn CredentialProvider>,
    retry: RetryPolicy,
    timeout: TimeoutPolicy,
    limits: ResourceLimits,
    max_inflight: usize,
    transport: Arc<dyn Transport>,
    sleeper: Arc<dyn Sleeper>,
    events: Arc<dyn EventSink>,
    rng_seed: u64,
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self {
            model: None,
            credential: Arc::new(EnvCredential::new(DEFAULT_ENV_VAR)),
            retry: RetryPolicy::default(),
            timeout: TimeoutPolicy::default(),
            limits: ResourceLimits::default(),
            max_inflight: DEFAULT_MAX_INFLIGHT,
            transport: default_transport(),
            sleeper: Arc::new(RealSleeper),
            events: Arc::new(NoopEventSink),
            rng_seed: DEFAULT_RNG_SEED,
        }
    }
}

impl ClientBuilder {
    /// Start a builder with a model already set.
    pub fn new(model: impl Into<ModelRef>) -> Self {
        Self::default().model(model)
    }

    /// Set the required model reference.
    pub fn model(mut self, model: impl Into<ModelRef>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the credential provider.
    pub fn credential(mut self, provider: impl CredentialProvider + 'static) -> Self {
        self.credential = Arc::new(provider);
        self
    }

    /// Set the retry policy.
    pub fn retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    /// Set the timeout policy.
    pub fn timeout(mut self, timeout: TimeoutPolicy) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the resource limits.
    pub fn limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Set the maximum number of concurrent in-flight calls.
    pub fn max_inflight(mut self, max_inflight: usize) -> Self {
        self.max_inflight = max_inflight;
        self
    }

    /// Set the transport. Tests use [`crate::transport::testing::MockTransport`].
    pub fn transport(mut self, transport: impl Transport + 'static) -> Self {
        self.transport = Arc::new(transport);
        self
    }

    /// Set the blocking sleeper (injectable for deterministic tests).
    pub fn sleeper(mut self, sleeper: impl Sleeper + 'static) -> Self {
        self.sleeper = Arc::new(sleeper);
        self
    }

    /// Set the metadata-only event sink.
    pub fn events(mut self, events: impl EventSink + 'static) -> Self {
        self.events = Arc::new(events);
        self
    }

    /// Set the deterministic jitter seed.
    pub fn rng_seed(mut self, seed: u64) -> Self {
        self.rng_seed = seed;
        self
    }

    /// Validate and build the client.
    pub fn build(self) -> Result<Client> {
        let model = self
            .model
            .ok_or_else(|| JevError::configuration_field("model", "a model is required"))?;
        self.limits.validate()?;
        self.timeout.validate()?;
        self.retry.validate()?;
        if self.max_inflight == 0 {
            return Err(JevError::configuration_field(
                "max_inflight",
                "max_inflight must be positive",
            ));
        }
        // Fail fast if the endpoint policy were ever bypassed at build time.
        endpoint_url(SYSTEM_ONE_PATH)?;
        Ok(Client {
            model,
            credential: self.credential,
            retry: self.retry,
            timeout: self.timeout,
            limits: self.limits,
            transport: self.transport,
            sleeper: self.sleeper,
            events: self.events,
            lifecycle: Lifecycle::new(self.max_inflight),
            rng: Mutex::new(self.rng_seed.max(1)),
        })
    }
}

/// A counting semaphore that also tracks the closed flag.
struct Lifecycle {
    state: Mutex<LifecycleState>,
    cv: Condvar,
    max_inflight: usize,
}

struct LifecycleState {
    closed: bool,
    active: usize,
}

impl Lifecycle {
    fn new(max_inflight: usize) -> Self {
        Self {
            state: Mutex::new(LifecycleState {
                closed: false,
                active: 0,
            }),
            cv: Condvar::new(),
            max_inflight,
        }
    }

    fn acquire(&self) -> Result<LifecycleGuard<'_>> {
        let mut state = self.state.lock().expect("lifecycle poisoned");
        while state.active >= self.max_inflight && !state.closed {
            state = self.cv.wait(state).expect("lifecycle poisoned");
        }
        if state.closed {
            return Err(JevError::Closed);
        }
        state.active += 1;
        Ok(LifecycleGuard { lifecycle: self })
    }

    fn release(&self) {
        let mut state = self.state.lock().expect("lifecycle poisoned");
        state.active = state.active.saturating_sub(1);
        self.cv.notify_all();
    }

    fn close(&self) {
        let mut state = self.state.lock().expect("lifecycle poisoned");
        state.closed = true;
        self.cv.notify_all();
        while state.active > 0 {
            state = self.cv.wait(state).expect("lifecycle poisoned");
        }
    }

    fn is_open(&self) -> bool {
        !self.state.lock().expect("lifecycle poisoned").closed
    }
}

struct LifecycleGuard<'a> {
    lifecycle: &'a Lifecycle,
}

impl Drop for LifecycleGuard<'_> {
    fn drop(&mut self) {
        self.lifecycle.release();
    }
}
