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
- `model`: the ModernBERT encoder + decision head — embedding + `embed_norm`,
  per-layer `attn_norm`, fused-QKV split, non-traditional RoPE with per-kind
  base, full/sliding masks with padded-query handling, GeGLU MLP, `final_norm`;
  the typed head (no RoPE, ReLU MLP), marker gather/clamp, scorer softmax,
  top-1/top-2/entropy pooling, and the action head. Both a host
  `forward_encoder_reference` / `forward_reference` and a tenferro
  `forward_encoder_tenferro` / `forward_tenferro` are provided, with synthetic
  parity tests. The forward uses the tanh GELU on both sides so they can be
  compared; an `exact_gelu` / `erf` port of `mlx_erf` is included for the
  eventual exact path.
- Remaining: concrete tokenizer (byte-level / Metaspace BPE), safetensors
  loading, `DecisionEngine` wiring, plans/workspaces.

### Phase 5 — `jeff-infer`

- `config`: supported Qwen3.5 `TextConfig` subset and `DecisionConfig`.
- `readout`: temperature-scaled softmax and the choice/noul/score formulas.
- `model`: the Qwen3.5 layer stack — embedding, input/post RMSNorm, full
  attention (GQA-expanded, split `q`/gate, Q/K RMSNorm, partial RoPE, causal
  mask, output projection) or Gated DeltaNet (`tenferro-gated-delta`), the
  SiLU-gated MLP, the final last-position RMSNorm, and the readout. Both a host
  `forward_reference` and a tenferro `forward_tenferro` are provided.
- `checkpoint`: a self-contained safetensors reader (no added dependency;
  `F32`/`F64`/`F16`/`BF16`) and a strict loader that maps `language_model.*`
  tensors plus `readout.safetensors` into `JeffWeights` — interleaved per-head
  `[query, gate]` split of the fused `q_proj`, consecutive GQA expansion,
  `a_decay = -exp(A_log)`, and the `(channels, 1, taps)` convolution layout.
- Real-fixture parity: against the `extern/JeffClient.jl` synthetic Qwen3.5
  fixture (lengths 1/3/63/64/65, batch 2, left padding), `forward_reference`
  matches the independent PyTorch logits to below `1e-4` (observed ~`5e-7`) and
  `forward_tenferro` to below `1e-3`. The tests skip when the fixture submodule
  is absent.
- Remaining: tokenizer, prepared-token `DecisionEngine` wiring, plans/workspaces.

### Phase 6 — `tenferro-gated-delta`

- Host reference: causal depthwise convolution + SiLU, `ops` helpers, and the
  recurrent Gated DeltaNet scan (`reference`) for one value head.
- Tenferro-backed chunked scan (`chunked`) using eager `matmul`/`dot_general`,
  `triangular_solve(unit_diagonal = true)`, `exp`, and the shared `rms_norm` /
  `silu` primitives.
- Full layer wrapper (`layer`): `qkv`/`z`/`a`/`b`/`out_proj` projections
  (tenferro-backed), causal convolution, Q/K L2 normalization with
  `1 / sqrt(key_dim)` Q scaling, `beta`/`decay` from `sigmoid`/`softplus`, the
  scan, and the output projection — with `GatedDeltaWeights` layout validation.
- Cross-formulation parity tests: the chunked scan and the full tenferro layer
  match the recurrent host reference, across chunk boundaries, single tokens,
  grouped key/value widths, mask holes, and strongly negative decay.
- Fixed against the Qwen3.5 fixture: the chunked state update computed the
  ending-key decay as `exp(sum).ln()`, which underflows to `-inf` once a
  64-token chunk's decay sum is very negative; it now uses the host cumulative
  values directly (`regression test with strong decay across a chunk
  boundary`). The Q scaling was also corrected from `sqrt(key_dim)` to
  `1 / sqrt(key_dim)` (the sign shows through the output-RMSNorm `eps`).
- Also the chunked effective-system `M = I + L` helper with unit-diagonal
  forward substitution.
- Remaining: prepared plans/workspaces, extension-op wiring, fused CPU
  convolution/normalization, and CUDA kernels.

## Blocked / needs external input

| Area | Blocker |
|---|---|
| Laya numerical parity (Phase 2) | No Laya checkpoint/fixture in this environment; validate against `extern/Laya.jl` once tokenizer/weights assets exist. (Jeff now has real-fixture parity in `native_fixture.rs`.) |
| Exact GELU (Laya parity) | tenferro has no `erf`; needs a `tenferro-infer` extension op porting `mlx_erf`. |
| Phase 4 / 7 CUDA | No CUDA hardware here; code can be written but not validated. |
| Phase 8 FP16/BF16 | Deferred by decision: tenferro's public dtype set lacks `F16`/`BF16` at the pinned revision. |
| Phase 10 Apple GPU | tenferro's WebGPU surface is effectively `dot_general` (F32/C32) plus transpose; most primitives are missing. |
| Tokenizer (Laya) | Requires the checkpoint's tokenizer assets. |

## Verification snapshot

- `cargo test --workspace`: decision-core 20, reference-data 4, tenferro-infer 13,
  laya-infer 27, jeff-infer 24, tenferro-gated-delta 10, jev-client 49
  (58 with `--features http`), bench-suite 1.
- `cargo clippy --workspace --all-targets -- -D warnings`: clean.
