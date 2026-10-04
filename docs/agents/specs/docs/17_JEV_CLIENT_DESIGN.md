# jev-client Design

**Status:** design proposal (implements the package's Jev requirements)  
**Date:** 2026-10-04  
**Source reference:** `extern/JevClient.jl` at commit `4836ef382ddb5eb32e9dbd7e539880486beecaf9` and its wire spec `extern/JevClient.jl/docs/agents/spec.md`  
**Related:** `02_SPECIFICATION.md` §6, `04_MODEL_MAPPING.md` §3, `07_SECURITY_LICENSE.md` §3–§5, `16_DECISION_CORE_DESIGN.md`

`jev-client` is the independent Rust client for the TypeSafe AI System One API
(Jev). It **must not depend on tenferro** (`02_SPECIFICATION.md` §6). It shares
only `decision-core` types with the local engines.

---

## 1. Scope and responsibilities

Owns:

- fixed TypeSafe endpoint policy
- HTTPS/TLS transport with explicit verification
- credential providers and redaction
- retries, timeouts, and resource limits
- JSON serialization of state/questions into the wire format
- strict response parsing and cross-field validation
- logging and observability (metadata only)

Does not own:

- tensor inference (Laya/Jeff)
- geff generation or streaming (out of scope)
- arbitrary `base_url`, proxies, or additional endpoints in 0.1

---

## 2. Crate layout and public API

```text
crates/jev-client/
├── src/
│   ├── lib.rs
│   ├── client.rs        # Client, system_one, list_models, lifecycle
│   ├── credentials.rs   # providers, redaction, zeroization
│   ├── models.rs        # PinnedModel, MovingAlias, ModelInfo, ModelList
│   ├── content.rs       # allowed JSON-compatible values
│   ├── questions.rs     # wire question construction
│   ├── serialization.rs # the single write path to wire JSON
│   ├── transport.rs     # TLS, no redirect, no proxy, size caps
│   ├── retry.rs         # retry state machine
│   ├── limits.rs        # ResourceLimits, TimeoutPolicy, RetryPolicy
│   ├── responses.rs     # strict parse + cross-field validation
│   ├── errors.rs        # error hierarchy
│   └── logging.rs       # metadata-only events + redaction
└── tests/
```

Public API (proposed), mirroring the Julia client:

```rust
pub struct Client { /* model, credential, policies, transport, inflight guard */ }

pub struct ClientBuilder { /* model required; credential/policies/limits/max_inflight */ }

impl Client {
    pub fn builder() -> ClientBuilder;
    pub fn system_one(&self, req: &SystemOneRequest) -> Result<SystemOneResponse, JevError>;
    pub fn list_models(&self) -> Result<ModelList, JevError>;
    pub fn is_open(&self) -> bool;
    pub fn close(&self);
}

pub enum ModelRef { Pinned(PinnedModel), MovingAlias(MovingAlias) }
```

A scoped `with_client` helper is provided. The request carries `state` and a
`QuestionSet`; the model comes from the client.

---

## 3. Endpoint policy

Hard-fixed in 0.1:

```text
scheme = https
host   = api.typesafe.ai
port   = 443
paths  = /v1/systemone (POST), /v1/models (GET)
```

- Never read `TYPESAFE_BASE_URL` or accept a URL from the public constructor.
- No userinfo, query, or fragment; no redirect following.
- This prevents configuration injection, SSRF, and credential misdirection.

---

## 4. Transport

- TLS certificate and hostname verification are mandatory and cannot be
  disabled through any public option; do not retry TLS errors.
- HTTP redirects (3xx) are `RedirectError`; `Authorization` is never forwarded.
- Proxy defaults to none; do not read `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY`
  implicitly.
- Only HTTP 200 is success.
- The underlying HTTP library's automatic retry and redirect are disabled; all
  retry logic lives in this crate's single state machine.
- Response bodies are read with a size cap: reject an oversized
  `Content-Length` before reading, and abort chunked bodies on the cumulative
  cap.
- Compression is not requested (`Accept-Encoding: identity`) and compressed
  responses are rejected by default.

Fixed request headers:

```http
Authorization: Bearer <redacted>
Content-Type: application/json; charset=utf-8
Accept: application/json
Accept-Encoding: identity
User-Agent: jev-client-rs/<version> rust/<version>
```

`Authorization`, `Content-Type`, `Accept`, and `Host` cannot be overridden by
the caller. No cookies, no request-body compression.

The transport is an injectable trait so tests use a `MockTransport` and no real
network; the internal request/response objects are never exposed publicly.

---

## 5. Credentials

```rust
pub trait CredentialProvider: Send + Sync { fn credential(&self) -> Result<Secret>; }
pub struct EnvCredential { /* reads env var at request time */ }
pub struct StaticCredential { /* owns bytes; zeroized on close */ }
pub struct CredentialCallback { /* invoked before each request */ }
```

- Validate the API key before sending: trim ASCII whitespace, reject empty,
  reject internal whitespace/control characters/non-ASCII, reject > 4 KiB, no
  assumed prefix or minimum length.
- Build the `Authorization` header immediately before sending; zeroize
  temporary buffers after use (best effort).
- Never place the key in a global constant, config file, precompile cache, query
  parameter, log, `Debug`/`Display`, exception, or tracing metadata.
- `Client` owns its credential provider; `close` closes it. `StaticCredential`
  zeroizes on close. `CredentialCallback` exceptions become redacted
  `CredentialError`s.
- All providers render as `<redacted>`.
- The secret type wraps bytes and only exposes them through an explicit,
  narrowly scoped accessor used to build the header.

---

## 6. Limits, timeouts, retries

```rust
pub struct ResourceLimits {
    pub max_request_bytes: usize,      // default 1 MiB
    pub max_response_bytes: usize,     // default 8 MiB
    pub max_error_body_bytes: usize,   // default 64 KiB
    pub max_json_depth: usize,         // default 32
    pub max_string_bytes: usize,       // default 1 MiB
    pub max_container_items: usize,    // default 100_000
    pub max_questions: usize,          // default 1024
    pub max_question_id_chars: usize,  // default 128
    pub max_model_id_bytes: usize,     // default 128
}

pub struct TimeoutPolicy { connect, first_byte, attempt, total }  // 5/15/30/60 s
pub struct RetryPolicy { max_retries, initial_delay, max_delay, total_budget } // 2/0.5/15/60 s
```

- All limits positive; enforced locally before serialization and after reading.
- Limits are not relaxed per call; build a new `Client` to change them.
- Timeouts: `connect <= attempt <= total`, `first_byte <= attempt`.
- Retry only HTTP 429 and 529, with exponential backoff and full jitter,
  honoring `Retry-After` (seconds or HTTP-date; reject invalid/negative/oversized).
- `max_retries` 0..=5; `total_budget <= 300 s`.
- Do not retry TLS errors, redirects, JSON/validation failures, or ambiguous I/O
  failures (the body may already have been sent).
- The semaphore wait counts against the total deadline.
- Do not hold the key or serialized body while waiting; rebuild both per attempt.

The clock and sleeper are injectable internally for deterministic tests.

---

## 7. Content model (no auto-serialization)

`state` and `instructions` accept only explicitly constructed allowed values:
`null`, `bool`, string, safe integer, finite float, array, and an object whose
keys are strings (with Symbol/duplicate key handling as appropriate in Rust:
duplicate keys rejected). Rejected: `NaN`/`Inf`, unbounded integers, arbitrary
structs, functions/tasks/IO/pointers, implicit byte-to-string coercion, cycles,
invalid UTF-8.

`Content` is a dedicated type, not a generic `serde_json::Value` passthrough, so
that user structs are never automatically expanded and secrets inside them are
never transmitted. If custom types are supported later, require an explicit
conversion and forbid generic reflection.

`state` top level is string/object/array only (not `null`/number/bool).
Insertion order is preserved for objects because it can affect results.

---

## 8. Request serialization

Single write path (`serialization.rs`):

```json
{
  "state": "string, object, or array",
  "model": "jev-1.13.0",
  "questions": {
    "question_id": {
      "type": "noul | choice | score",
      "instructions": "string, object, or array",
      "criteria": "type-dependent"
    }
  }
}
```

- Noul wire keys are exactly `true`/`false`.
- Choice criteria is an object of candidate ID -> description.
- Score criteria is an ordered array of strings; the 0-origin index is the wire
  value.
- The request body is serialized per attempt and checked against
  `max_request_bytes`.

---

## 9. Response validation

Treat server responses as untrusted.

- `Content-Type` is `application/json` or `application/*+json`.
- Valid UTF-8; object top level; inspect depth and duplicate keys before
  materializing.
- `model` non-empty; `usage.input_tokens`/`output_tokens` non-negative integers.
- Answer key set exactly equals the sent question IDs; each answer `type`
  matches its question type; unknown answer types are errors, not skipped.
- Unknown top-level fields may be ignored for forward compatibility but never
  retained or exposed.
- `NoulAnswer.noul`: finite, `0..=1`.
- `ChoiceAnswer`: probability keys equal the sent candidates; all finite in
  `0..=1`; sum `1 ± 1e-4`; `choice` is a known candidate and within `1e-6` of
  the max; confidence finite in `0..=1`.
- `ScoreAnswer`: legend/probabilities as array or 0-origin keyed object;
  canonical decimal indices covering `0..n-1`; lengths match criteria; finite in
  `0..=1`; sum `1 ± 1e-4`; `score` in `0..=n-1` and within `1e-4` of
  `sum(i * p[i])`; confidence finite in `0..=1`.
- No raw response body or escape hatch is exposed.

---

## 10. Errors, logging, lifecycle

Error hierarchy mirrors the Julia client: configuration, transport, protocol,
API, retry-budget, and concurrency errors, each carrying only safe context
(message, status, request ID, retry-after, field path, attempt count). Remote
bodies are never stored; remote 422 details are normalized to field path,
error code, and a short safe message.

Logging is metadata-only: operation, version, requested/returned model, question
counts by type, byte counts, status, latency, retry count/reason, request ID,
token usage. Never log keys, headers, `state`, `instructions`, `criteria`,
bodies, candidate IDs, or question IDs. Suppress the underlying HTTP library's
wire-level debug output during calls.

Lifecycle: `Client` is shareable across tasks; a per-client connection pool and
an in-flight semaphore (`max_inflight`, default 8) are used; counters are
lock/atomic protected. `close` stops new requests and waits for in-flight ones
to finish without cancelling them; post-close calls return `ClosedClientError`.

---

## 11. decision-core integration

`jev-client` shares `decision-core` question/answer/response types. The wire
serialization is owned by this crate, but the public request/answer types come
from `decision-core` (`16_DECISION_CORE_DESIGN.md`). The client can implement the
optional `DecisionEngine` trait as a remote adapter, but Jev is not a local
tensor engine and has no CPU/GPU inference.

Model references are typed: `PinnedModel` (recommended for reproducible
production) versus `MovingAlias` (warns once, not reproducible). Responses
record the returned versioned `model`.

---

## 12. Model output safety (documentation requirement)

The crate documentation must state that Jev output is probability-bearing
untrusted data, never authorization, instructions, or evidence. Choice output
must match an application-defined allowlist and map deterministically to
handlers; never interpolate it into shell/SQL/paths/URLs/dynamic dispatch.
High-impact decisions require deterministic rules or human review. These are
documentation/API-guidance requirements, not runtime guarantees.

---

## 13. Testing and acceptance

Unit and security tests:

- question/content validation, size/depth/cycle limits, duplicate keys
- credential trim/rejection and sentinel-secret leak checks across stdout,
  logs, `Debug`/`Display`, errors, and test output
- fixed headers and endpoint; redirect not followed (including same-origin);
  proxy env vars ignored; TLS failure not retried
- only 429/529 retried; oversized `Content-Length` and chunked bodies capped;
  gzip rejected; malformed UTF-8 / duplicate keys / deep JSON rejected
- response cross-field validation matrix; unknown answer type rejected
- deterministic retry schedule, timeout/deadline, `close` lifecycle,
  `max_inflight` concurrency

Acceptance for 0.1 follows `extern/JevClient.jl/docs/agents/spec.md` §24:
secrets never appear in output, bodies never logged, the key cannot be sent
anywhere but `api.typesafe.ai:443`, redirects are not followed, TLS cannot be
disabled, no implicit proxy, size caps exist, only 429/529 retried, ambiguous
I/O not retried, all answers validated, no raw response retained, no arbitrary
struct auto-serialized, pinned-model example present, and the documentation
carries the adversarial/high-impact/privacy warnings.

---

## 14. Open questions

- [ ] Rust HTTP/TLS stack selection (e.g. `reqwest` + `rustls`) while keeping
      redirects/proxies/auto-retry disabled and body caps enforced.
- [ ] How to expose `Content` ergonomically without enabling arbitrary struct
      serialization.
- [ ] Whether `Client` should implement `DecisionEngine` or only expose
      `system_one`.
- [ ] Streaming body cap implementation for chunked responses.
- [ ] Whether model-output safety guidance lives only in docs or also in a
      typed `Choice` allowlist helper.
- [ ] Optional future `base_url` support behind a separately specified,
      auditable credential-scoping mechanism (explicitly out of 0.1).
