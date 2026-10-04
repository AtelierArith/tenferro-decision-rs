# typed-decision-rs design package

**Status:** design / specification draft  
**Date:** 2026-10-04

This archive consolidates the design discussion for porting the inference functionality of:

- `AtelierArith/Laya.jl`
- `AtelierArith/JeffClient.jl`
- `AtelierArith/JevClient.jl`

to Rust, using `tensor4all/tenferro-rs` as the numerical/runtime substrate where appropriate.

The project is explicitly **inference-first** and **performance-first**.

## Core decision

The proposed Rust workspace should not try to become a general neural-network framework.

Instead:

- `Laya` becomes a highly optimized inference engine on top of tenferro.
- `Jeff` becomes a highly optimized Qwen3.5 + Gated DeltaNet inference engine on top of tenferro.
- `JevClient` becomes an independent Rust HTTP client and must **not** depend on tenferro.
- CPU and CUDA GPU inference are primary targets.
- Apple GPU / Metal is a follow-up target, because tenferro's WebGPU/Metal operation coverage is currently narrower than its CPU/CUDA path.
- Autodiff, training, optimizers, and generic model-authoring APIs are intentionally excluded.

## Recommended workspace

```text
typed-decision-rs/
├── crates/
│   ├── decision-core/
│   ├── tenferro-infer/
│   ├── tenferro-gated-delta/
│   ├── laya-infer/
│   ├── jeff-infer/
│   └── jev-client/
├── benchmarks/
├── tests/
└── tools/
```

## Documents

- `docs/01_DESIGN.md` — architecture and major design decisions
- `docs/02_SPECIFICATION.md` — normative functional/non-functional specification
- `docs/03_CPU_GPU_INFERENCE.md` — CPU/CUDA execution model and optimization policy
- `docs/04_MODEL_MAPPING.md` — Julia → Rust/tenferro mapping for Laya, Jeff, Jev
- `docs/05_TESTING_BENCHMARKS.md` — correctness, numerical parity, benchmark and profiling requirements
- `docs/06_ROADMAP.md` — implementation sequence and release gates
- `docs/07_SECURITY_LICENSE.md` — security, checkpoint, credential, and licensing requirements
- `docs/08_SOURCE_SNAPSHOT.md` — source repositories and observed implementation facts used for this design
- `docs/09_JEFFCLIENT_ANALYSIS.md` — detailed analysis of the JeffClient.jl native inference path and its Rust/tenferro porting implications
- `docs/10_DECISION_ABSTRACTION.md` — proposed `decision-core` surface and `DecisionEngine` seam for Laya-first integration of Jeff and Jev
- `docs/11_TENFERRO_API_SURVEY.md` — tenferro-rs API coverage survey for the Laya/Jeff/tenferro-gated-delta crates at revision `471c4278`
- `docs/12_TENFERRO_GATED_DELTA.md` — design of the `tenferro-gated-delta` crate (causal conv, reference/chunked/recurrent Delta, CPU/CUDA, extension-op integration)
- `docs/13_LAYA_INFER_DESIGN.md` — design of the `laya-infer` crate (ModernBERT encoder, decision head, tokenizer, prompt, calibration)
- `docs/14_JEFF_INFER_DESIGN.md` — design of the `jeff-infer` crate (Qwen3.5 hybrid stack, readout, prepared-token inference)
- `docs/15_TENFERRO_INFER_DESIGN.md` — design of the shared `tenferro-infer` primitive crate (norm, activations, softmax, RoPE, attention)
- `docs/16_DECISION_CORE_DESIGN.md` — design of the backend-independent `decision-core` crate and its `DecisionEngine` trait
- `docs/17_JEV_CLIENT_DESIGN.md` — design of the independent `jev-client` TypeSafe API client (transport, credentials, limits, validation)
- `docs/18_IMPLEMENTATION_STATUS.md` — live status of the Rust implementation against the roadmap (completed phases, blockers)
- `docs/19_TENFERRO_FEEDBACK.md` — tenferro-rs feedback filed as upstream issues

## Recommended first milestone

The first serious milestone is **Laya CPU optimized inference**, not merely a tiny demo.

Completion means:

- real Laya checkpoint
- pure Rust runtime
- tokenizer included
- tenferro-backed inference
- Float32 correctness against Laya.jl
- all decision types
- preallocated steady-state workspace
- no avoidable forward allocations
- reproducible latency benchmark
- profiling report

CUDA follows immediately after that.
