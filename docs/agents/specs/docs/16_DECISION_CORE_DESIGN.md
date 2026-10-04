# decision-core Design

**Status:** design proposal (concretizes `10_DECISION_ABSTRACTION.md`)  
**Date:** 2026-10-04  
**Related:** `01_DESIGN.md` §4, `02_SPECIFICATION.md`, `04_MODEL_MAPPING.md` §3–§4, `10_DECISION_ABSTRACTION.md`, `13_LAYA_INFER_DESIGN.md`, `14_JEFF_INFER_DESIGN.md`, `17_JEV_CLIENT_DESIGN.md`

`decision-core` owns the backend-independent typed-decision data and the
optional `DecisionEngine` seam shared by Laya, Jeff and Jev. It **must not
depend on tenferro** (`01_DESIGN.md` §4). Its types are the stable surface the
application codes against, so adding a later engine does not change them.

---

## 1. Scope and responsibilities

Owns:

- question types and validation
- answer types
- request/response envelopes
- usage metadata
- error categories
- the optional `DecisionEngine` trait

Does not own:

- any tensor/backend type
- model execution (engine crates)
- HTTP transport (in `jev-client`)
- prompt formatting or JSON wire serialization (engine/client specific)

---

## 2. Crate layout and public API

```text
crates/decision-core/
├── src/
│   ├── lib.rs
│   ├── question.rs     # Question, ChoiceQuestion, NoulQuestion, ScoreQuestion
│   ├── answer.rs       # Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer
│   ├── response.rs     # SystemOneResponse, Usage
│   ├── error.rs        # DecisionError
│   └── engine.rs       # DecisionEngine trait
└── tests/
```

The crate is pure Rust with `serde` support (feature-gated) and no I/O, no
network, no tensors.

---

## 3. Types

### 3.1 Questions

```rust
pub struct QuestionSet { pub questions: Vec<(String, Question)> }  // order preserved

pub enum Question {
    Choice(ChoiceQuestion),
    Noul(NoulQuestion),
    Score(ScoreQuestion),
}

pub struct ChoiceQuestion {
    pub instructions: Content,              // string or JSON value
    pub criteria: Vec<(String, Content)>,   // ordered label => description
}

pub struct NoulQuestion {
    pub instructions: Content,
    pub criteria: Option<NoulCriteria>,     // { true: Content, false: Content }
}

pub struct ScoreQuestion {
    pub instructions: Content,
    pub criteria: Vec<String>,              // ordered, non-empty labels
}
```

Validation mirrors the reference behavior:

- QuestionSet: 1..=1024 questions; IDs 1..=128 Unicode scalars, non-empty, no
  control characters or surrounding whitespace, unique; insertion order
  preserved because evaluation order can matter.
- Choice: 2..=255 candidates; unique, non-empty IDs; descriptions are
  `Content` (string, object, array, or none).
- Noul: at least one of true/false criteria.
- Score: 2..=10 ordered non-empty labels.
- `instructions` is required and non-empty.

`Content` represents the allowed JSON-compatible values (`null`, `bool`,
`string`, finite number, array, object with string keys, no NaN/Inf, no cycles).
It is deliberately **not** an arbitrary serde value so that custom structs are
never auto-serialized (see `17_JEV_CLIENT_DESIGN.md` §7).

### 3.2 Answers

```rust
pub enum Answer {
    Choice(ChoiceAnswer),
    Noul(NoulAnswer),
    Score(ScoreAnswer),
}

pub struct ChoiceAnswer {
    pub choice: String,
    pub probabilities: Vec<(String, f64)>,   // insertion order = criteria order
    pub confidence: f64,
}

pub struct NoulAnswer { pub noul: f64 }

pub struct ScoreAnswer {
    pub score: f64,
    pub legend: Vec<String>,
    pub probabilities: Vec<f64>,
    pub confidence: f64,
}
```

These are the shapes returned by `13_LAYA_INFER_DESIGN.md` §8 and
`17_JEV_CLIENT_DESIGN.md` §9. Calibration semantics differ per engine but the
result type is common.

### 3.3 Response and usage

```rust
pub struct SystemOneResponse {
    pub model: String,
    pub answers: Vec<(String, Answer)>,       // question id => answer, ordered
    pub usage: Usage,
    pub request_id: Option<String>,
}

pub struct Usage { pub input_tokens: u64, pub output_tokens: u64 }
```

Accessors (`answer(response, id)`, `response[id]`) are the stable lookup API;
the internal storage type is not part of the contract.

### 3.4 Errors

```rust
pub enum DecisionError {
    InvalidInput { field: Option<String>, message: String },
    UnsupportedConfig { message: String },
    Backend { source: Box<dyn Error + Send + Sync> },
    Transport { source: Box<dyn Error + Send + Sync> },
    Remote { status: u16, request_id: Option<String> },
    // ...
}
```

Rules:

- categories live here; engine/client specifics are carried as opaque sources
- typed, no string-only errors (`02_SPECIFICATION.md` §12)
- no secrets, request bodies, or model inputs in error context
  (`07_SECURITY_LICENSE.md` §3)

---

## 4. The `DecisionEngine` trait

```rust
pub trait DecisionEngine {
    type Error: std::error::Error + Send + Sync + 'static;

    fn system_one(
        &mut self,
        state: &State,
        questions: &QuestionSet,
    ) -> Result<Vec<Answer>, Self::Error>;
}
```

Design notes (see `10_DECISION_ABSTRACTION.md` §4 for the rationale):

- Introduced only after Laya execution is stable; the trait is an adapter, not
  the primary engine API.
- `State` is the input representation. Because Laya consumes text, Jeff consumes
  prepared tokens, and Jev serializes a JSON value, the design chooses one of:
  - (a) an opaque enum with engine-specific variants, or
  - (b) a higher-level request with engine-owned preparation.
  The choice is deferred; the trait stays thin either way.
- Returns `decision-core` answers, not transport-shaped structs.
- `&mut self` matches the Model/Context pattern (mutable workspace).

---

## 5. Dependency and layering rules

- `decision-core` → no tensor/runtime/network dependency.
- `laya-infer`, `jeff-infer`, `jev-client` → `decision-core`.
- **`jev-client` must not depend on tenferro** (`02_SPECIFICATION.md` §6).
- Optional `serde` support only; the engine/client crates own their wire formats.

```text
LayaContext ─┐
JeffContext ─┼─ implement ─→ DecisionEngine (decision-core)
JevClient   ─┘
```

---

## 6. Testing

- constructor/validation matrix for every question type
- accepted/rejected `Content` values (NaN/Inf, cycles, depth, duplicate keys,
  arbitrary structs)
- answer/response shape and ordering
- `DecisionEngine` adapter tests per engine (additive, no engine rewrite)
- deterministic serialization of `Content` when `serde` is enabled

---

## 7. Open questions

- [ ] `State` representation (per-engine enum vs higher-level request).
- [ ] Whether calibration helpers live here or in each engine.
- [ ] Whether `SystemOneResponse` should be generic over the engine metadata.
- [ ] Feature-gating strategy for `serde` and for any schema (JSON) helpers.
- [ ] Whether question IDs use `String` everywhere or a validated newtype.
