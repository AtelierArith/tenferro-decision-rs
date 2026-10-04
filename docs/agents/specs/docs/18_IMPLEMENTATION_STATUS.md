# Implementation Status

Tracks what has been built in the Rust workspace against `06_ROADMAP.md`.
Updated as work lands.

## Completed

### Phase 0 — Workspace and baseline

- Virtual Cargo workspace (`crates/*`), shared dependency block.
- `decision-core`: typed questions/answers, `Content`, `State`, `DecisionError`,
  and the `DecisionEngine` trait. No tenferro dependency.
- tenferro CPU wiring pinned to the surveyed revision
  `471c4278dbc5a955b8a1664789c37e5d07e743cb` (the crates.io `0.7.1` release
  differs in the session-entry API).
- `reference-data` fixture format + loader; `fixtures/`.
- `bench-suite` criterion skeleton with build/machine metadata capture.
- CI: `cargo fmt --check`, `cargo clippy --workspace --all-targets -D warnings`,
  `cargo test --workspace`, and the `decision-core` `serde` feature.

### Phase 1 — `tenferro-infer` reference primitives

Implemented on tenferro's eager runtime (`tenferro-ad`), each with an
independent numerical test:

- `linear` (`dot_general`), `embedding` (`gather`)
- `layer_norm`, `rms_norm` (centered / non-centered)
- `sigmoid`, `silu`, `gated_silu`, `gelu` (tanh), `geglu`
- `softmax`, `masked_softmax`
- `rope_modernbert`, `rope_qwen_partial`
- reference multi-head `attention` with optional mask

Known gap: tenferro has no `erf`, so `gelu` uses the tanh approximation. The
exact erf-based form that Laya's MLX kernels use requires an extension op.

### Phase 9 — `jev-client`

Independent TypeSafe System One client (no tenferro dependency), with a
`Transport` seam and a `testing::MockTransport` so tests never touch the
network:

- fixed endpoint policy, no base URL / proxy / redirects
- credential providers with validation, redaction, and best-effort zeroization
- `ResourceLimits`, `TimeoutPolicy`, `RetryPolicy`; retry only 429/529 with an
  injectable sleeper
- single ordered wire-serialization path and strict response validation
- typed `JevError` hierarchy and metadata-only `Event` sink

The concrete `reqwest` + `rustls` transport is implemented behind the
non-default `http` feature (`ReqwestTransport`): TLS verification on, redirects
and proxies disabled, explicit no-retry, and bounded response reads. Default
builds stay dependency-light and use `UnsupportedTransport`; hermetic tests run
without the feature.

Still deferred: HTTP-date `Retry-After`, an injectable monotonic clock, a
`zeroize`-backed secret, and OS-trust-store roots (the `http` feature uses
bundled webpki roots).

## In progress (checkpoint-independent groundwork)

### Phase 2 — `laya-infer`

- `config`: `EncoderConfig` / `AgentConfig` parsing and validation.
- `calibration`: temperature buckets, clamping, entropy confidence, and
  choice/score/noul answer construction plus the action probability.
- `prompt`: Python-compatible JSON (`py_float` / `py_json_content`), state and
  option rendering, and `build_prefix` / `build_sequence` over a `Tokenizer`
  trait, ported from `Laya.jl` `prompt.jl`.
- Remaining: concrete tokenizer (byte-level / Metaspace BPE), safetensors
  loading, ModernBERT + decision-head forward, `DecisionEngine` wiring.

### Phase 5 — `jeff-infer`

- `config`: supported Qwen3.5 `TextConfig` subset and `DecisionConfig`.
- `readout`: temperature-scaled softmax and the choice/noul/score formulas.
- `model`: the Qwen3.5 layer stack — embedding, input/post RMSNorm, full
  attention (GQA-expanded, split `q`/gate, Q/K RMSNorm, partial RoPE, causal
  mask, output projection) or Gated DeltaNet (`tenferro-gated-delta`), the
  SiLU-gated MLP, the final last-position RMSNorm, and the readout. Both a host
  `forward_reference` and a tenferro `forward_tenferro` are provided.
- Parity tests: the tenferro forward matches the host reference for a mixed
  stack (DeltaNet + full attention) and for single-kind stacks, including mask
  holes.
- Remaining: safetensors checkpoint loading, tokenizer, prepared-token
  `DecisionEngine` wiring, plans/workspaces.

### Phase 6 — `tenferro-gated-delta`

- Host reference: causal depthwise convolution + SiLU, `ops` helpers, and the
  recurrent Gated DeltaNet scan (`reference`) for one value head.
- Tenferro-backed chunked scan (`chunked`) using eager `matmul`/`dot_general`,
  `triangular_solve(unit_diagonal = true)`, `exp`, and the shared `rms_norm` /
  `silu` primitives.
- Full layer wrapper (`layer`): `qkv`/`z`/`a`/`b`/`out_proj` projections
  (tenferro-backed), causal convolution, Q/K L2 normalization with `sqrt(key_dim)`
  Q scaling, `beta`/`decay` from `sigmoid`/`softplus`, the scan, and the output
  projection — with `GatedDeltaWeights` layout validation.
- Cross-formulation parity tests: the chunked scan and the full tenferro layer
  match the recurrent host reference, across chunk boundaries, single tokens,
  grouped key/value widths, and mask holes.
- Also the chunked effective-system `M = I + L` helper with unit-diagonal
  forward substitution.
- Remaining: prepared plans/workspaces, extension-op wiring, fused CPU
  convolution/normalization, and CUDA kernels.

## Blocked / needs external input

| Area | Blocker |
|---|---|
| Laya/Jeff numerical parity (Phases 2, 5) | No real checkpoints available in this environment; validate against `extern/Laya.jl` / `extern/JeffClient.jl` once checkpoints/fixtures exist. |
| Exact GELU (Laya parity) | tenferro has no `erf`; needs a `tenferro-infer` extension op porting `mlx_erf`. |
| Phase 4 / 7 CUDA | No CUDA hardware here; code can be written but not validated. |
| Phase 8 FP16/BF16 | Deferred by decision: tenferro's public dtype set lacks `F16`/`BF16` at the pinned revision. |
| Phase 10 Apple GPU | tenferro's WebGPU surface is effectively `dot_general` (F32/C32) plus transpose; most primitives are missing. |
| Tokenizer (Laya) | Requires the checkpoint's tokenizer assets. |

## Verification snapshot

- `cargo test --workspace`: decision-core 20, reference-data 4, tenferro-infer 14,
  laya-infer 22, jeff-infer 16, tenferro-gated-delta 9, jev-client 49
  (58 with `--features http`), bench-suite 1.
- `cargo clippy --workspace --all-targets -- -D warnings`: clean.
