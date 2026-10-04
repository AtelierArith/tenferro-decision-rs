# Incremental Integration and Shared Decision Abstraction

**Status:** design proposal (extends the normative package)  
**Date:** 2026-10-04  
**Basis:** `01_DESIGN.md` §2, §4, §5, §10; `02_SPECIFICATION.md` §6; `04_MODEL_MAPPING.md` §3, §4, §5

This document makes explicit how the package supports a **Laya-first**
implementation that can later absorb **Jeff** and then **Jev** without changing
the application-facing surface. It does not replace the normative documents; it
records the seams, the proposed shared types, and the guardrails that keep the
later integration cheap.

---

## 1. Goal

Ship Laya CPU first, then Laya CUDA, and only afterwards add Jeff and Jev.
Adding a later engine must not:

- change the types the application already uses,
- force Laya to link Jeff or Jev code,
- require rewriting the Laya execution path, or
- couple any local tensor engine to the remote client.

The package already anticipates this: `decision-core` is backend-independent,
`laya-infer` / `jeff-infer` / `jev-client` are sibling engines, and
`DecisionEngine` is deliberately deferred.

---

## 2. Layering and dependency rules

```text
                         Application
                              │
                        decision-core
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
        laya-infer        jeff-infer        jev-client
            │                 │                 │
            └────────┬────────┘             HTTPS (TypeSafe API)
                     │
               tenferro-infer
                     │
            tenferro-gated-delta          (Jeff only)
                     │
                 tenferro-rs
                     │
              CPU / CUDA / later Apple
```

Normative dependency rules (already in the package):

- `decision-core` **MUST NOT** depend on tenferro (`01_DESIGN.md` §4).
- `jev-client` **MUST NOT** depend on tenferro (`01_DESIGN.md` §4,
  `02_SPECIFICATION.md` §6).
- `tenferro-gated-delta` is needed by `jeff-infer` only; `laya-infer` must not
  depend on it.
- `tenferro-infer` holds only primitives shared by Laya and Jeff; it is not a
  generic NN framework (`01_DESIGN.md` §4).

Practical consequence: build features and optional dependencies so that a
Laya-only binary does not link `jeff-infer`, `tenferro-gated-delta`, or
`jev-client`. The reverse is not required (a Jeff or Jev binary may reuse
`tenferro-infer` / `decision-core`).

---

## 3. `decision-core` surface

`decision-core` owns backend-independent typed-decision data. The package lists
the concepts; the concrete Rust shapes below are a **proposal** to freeze
before Laya's public API stabilizes.

### 3.1 Questions

```rust
pub struct QuestionSet {
    pub questions: Vec<Question>,
    // order is significant and MUST be preserved
}

pub enum Question {
    Choice(ChoiceQuestion),
    Noul(NoulQuestion),
    Score(ScoreQuestion),
}

pub struct ChoiceQuestion {
    /// Ordered (key, description) pairs; order matches checkpoint columns.
    pub criteria: Vec<(String, String)>,
    pub instructions: String,
}

pub struct NoulQuestion {
    pub instructions: String,
}

pub struct ScoreQuestion {
    pub criteria: Vec<String>,
    pub instructions: String,
}
```

Rules preserved from the reference behavior: option order is authoritative and
deterministic; choice keys are unique; score has 2..10 levels; choice has up to
the checkpoint's trained option limit.

### 3.2 Answers

```rust
pub enum Answer {
    Choice(ChoiceAnswer),
    Noul(NoulAnswer),
    Score(ScoreAnswer),
}

pub struct ChoiceAnswer {
    pub probabilities: Vec<(String, f64)>,
    pub choice: String,
    pub confidence: f64,
}

pub struct NoulAnswer {
    /// Probability of the "true" option.
    pub noul: f64,
}

pub struct ScoreAnswer {
    pub probabilities: Vec<(String, f64)>,
    pub legend: Vec<(String, String)>,
    pub score: f64,
    pub confidence: f64,
}
```

These are pure data and share no backend type. Calibration (temperature,
zero-based score, confidence formulas) belongs to the engine, not to the
transport or to `decision-core`, but the *result* shape is common.

### 3.3 Errors and usage

```rust
pub enum DecisionError {
    InvalidInput { .. },
    UnsupportedConfig { .. },
    Backend { .. },      // wraps an engine-specific error as an opaque source
    Transport { .. },    // jev-client only
    // ...
}

pub struct Usage {
    // tokens, sequence length, device, timing buckets, ...
}
```

`decision-core` defines the categories; each engine supplies concrete details
via `source()` / structured fields. No engine may leak its backend types into
the enum.

---

## 4. `DecisionEngine` trait (proposed, deferred)

The package sketches (`04_MODEL_MAPPING.md` §4):

```rust
pub trait DecisionEngine {
    type Error;
    fn system_one(
        &mut self,
        state: &State,
        questions: &QuestionSet,
    ) -> Result<SystemOneResponse, Self::Error>;
}
```

Guidance from the package: introduce this **only after Laya/Jeff execution is
stable**, and do not let it distort local implementations.

Refinement proposal to consider before adopting:

- Keep the trait **thin and optional**. Laya's own API is the source of truth
  during v0.1; the trait is an adapter over it, not the primary API.
- The `State` type is the weak point: Laya consumes natural-language state,
  Jeff consumes prepared token IDs and masks, Jev consumes serialized state.
  Either:
  - (a) `State` is an opaque enum with per-engine variants, or
  - (b) the trait is defined over a higher-level request and each engine owns
    its own input preparation behind the trait.
  Choose (a) if the application wants one call site; choose (b) if engines
  should own tokenization/prompt building.
- Return `Vec<Answer>` (or an ordered response containing them) rather than a
  transport-shaped struct, so the trait stays independent of the Jev wire
  format.
- `&mut self` is consistent with the Model/Context pattern (workspace is
  mutable). Whether the trait takes `&self` plus an explicit context is an
  open decision (see §8).

The trait lives in `decision-core` (no tenferro import) and is implemented by
each engine crate.

---

## 5. Model / Context pattern (reused for Jeff)

`01_DESIGN.md` §5 defines the template and states "The same pattern should be
usable for Jeff."

```rust
pub struct LayaModel  { weights: Arc<LayaWeights>,  plan: Arc<LayaPlan> }
pub struct LayaContext{ model: Arc<LayaModel>,      workspace: LayaWorkspace }

// later, identical shape
pub struct JeffModel  { weights: Arc<JeffWeights>,  plan: Arc<JeffPlan> }
pub struct JeffContext{ model: Arc<JeffModel>,      workspace: JeffWorkspace }
```

Why this matters for later integration:

- Immutable weights can be shared; workspaces are lock-free per request.
- CUDA streams become context-owned without a Laya/Jeff API change.
- Allocation lifetime is explicit, matching the package's ownership rule
  (`01_DESIGN.md` §3.5) and the hot-path/workspace requirements
  (`02_SPECIFICATION.md` §7, §8).

A `DecisionEngine` impl wraps a `Context` (`&mut self`).

---

## 6. Shared primitives versus engine-specific work

- `tenferro-infer` (Laya + Jeff): prepared GEMM, embedding gather, LayerNorm,
  RMSNorm, GELU/GeGLU, SiLU, softmax, masked softmax, RoPE, residual+norm
  fusion, reference attention, inference layout helpers.
- `tenferro-gated-delta` (Jeff only): causal depthwise convolution, recurrent
  delta, chunked/reference delta, CPU-optimized and CUDA-specialized kernels.
- Laya-specific work stays in `laya-infer`: checkpoint loading, tokenizer,
  prompt building, ModernBERT, decision head, calibration.

The package forbids the "one giant model kernel" (`01_DESIGN.md` §10):
granularity is *reusable inference primitive + model-specific plan*. This is
what lets Jeff reuse Laya's primitives without inheriting Laya's structure.

---

## 7. Jev boundary

Jev is not a local tensor engine (`04_MODEL_MAPPING.md` §3,
`08_SOURCE_SNAPSHOT.md`). It shares only `decision-core`:

```text
QuestionSet (decision-core) ──serialize──> jev-client ──HTTPS──> TypeSafe API
                                              │
                                        response validation
                                              │
                                     Answer (decision-core)
```

- `jev-client` owns TypeSafe transport, credentials, retries/timeouts,
  resource limits, response parsing, endpoint policy.
- It may be developed in parallel (Phase 9) and must never pull tenferro.
- Security requirements (`07_SECURITY_LICENSE.md` §3–§5) apply to this crate.

---

## 8. Recommended integration sequence

Aligned with the package roadmap, with the abstraction checkpoints called out:

1. **`decision-core` types frozen** for questions/answers/errors/usage
   (author Laya's public API against these from the start).
2. **`tenferro-infer` reference primitives.**
3. **Laya CPU correctness** (v0.1 foundation).
4. **Laya CPU optimization** — first production milestone (v0.1).
5. **Laya CUDA** (v0.2).
6. **Introduce `DecisionEngine`** now that Laya execution is stable; implement
   it for `LayaContext` and confirm it is additive (no Laya rewrite).
7. **Jeff CPU reference**, then **Gated Delta optimization**, then **Jeff CUDA**
   (v0.3, v0.4), implementing the same trait.
8. **Jev Rust client** (v0.5, independent).
9. **Apple GPU** after CPU/CUDA.

Checkpoint 6 is the only sequencing change this document adds: it makes the
previously implicit "wait until Laya/Jeff are stable" explicit, and validates
the seam before Jeff exists.

---

## 9. Guardrails that keep later integration cheap

Do:

- Keep every backend type behind a crate boundary; `decision-core` stays pure.
- Keep the Model/Context split; expose `Context` creation and a single decide
  entry point per engine.
- Return `decision-core` answer types, not engine-internal structs.
- Gate `jeff-infer` / `tenferro-gated-delta` / `jev-client` behind optional
  dependencies and features so Laya does not link them.
- Type all errors; no silent fallback (`02_SPECIFICATION.md` §12).

Do not:

- Put tenferro types in `decision-core` or `jev-client`.
- Make `DecisionEngine` the primary Laya API before Laya is stable.
- Collapse all engines into one opaque kernel or one shared backend cache.
- Let `tenferro-infer` grow model-specific logic that only Jeff uses.
- Assume a common `State` representation without deciding §4's (a)/(b).

---

## 10. Open decisions

- [ ] `State` representation for `DecisionEngine`: per-engine enum vs
      higher-level request with engine-owned preparation.
- [ ] Trait receiver: `&mut self` vs `&self` + explicit context.
- [ ] Batch semantics: one question vs ordered `QuestionSet` vs batched states.
- [ ] Whether `Usage`/timing is part of the trait response or separate.
- [ ] Where calibration lives (engine crate vs `decision-core` helper) so all
      three engines stay consistent.
- [ ] Crate layout for optional features that keep Laya binary minimal.
- [ ] Whether `DecisionEngine` is generic over input or uses an associated
      input type.
