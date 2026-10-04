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
- `safetensors-io`: a shared, dependency-light safetensors reader
  (`F32`/`F64`/`F16`/`BF16`) used by both engines' checkpoint loaders.
- `tenferro-ext`: a self-hosted tenferro extension op (`erf`, the same
  mechanism `tenferro-linalg`/`tenferro-fft` use) plus the exact erf-based GELU
  on the eager session; the `f32` kernel ports the MLX `erff` sequence. Used by
  Laya's forward.
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

### Model acquisition — `hf-fetch`

- `crates/hf-fetch` downloads production checkpoints from the Hugging Face Hub
  with the Julia-compatible cache layout (`models--<org>--<name>/snapshots/
  <commit>`, `refs/<revision>`) and environment variables (`HF_HUB_CACHE`,
  `HF_HOME`, `HF_ENDPOINT`, `HF_TOKEN`, `HF_HUB_OFFLINE`).
- Presets: `laya` → `convaiinnovations/laya`@`main` (matching `Laya.load`);
  `jeff` → `mstrasser/Jeff-Qwen3.5-0.8B`@`0f212b3e72acb4dde3f7da61e925d6ab7f819990`
  (the pin in `extern/JeffClient.jl/docs/src/models.md`). `CheckpointSpec`
  selects only the checkpoint files, with `Exact`/`Prefix`/`Glob` rules and a
  `required` completeness check. Downloads stage into a temp dir and move into
  the snapshot atomically.
- `hf-fetch` is a separate crate with no tenferro dependency; it returns a
  local snapshot directory for `LayaEngine::load` / `load_checkpoint`.
- CLI `hf-fetch <preset|org/name> [--revision …] [--subfolder …] [--offline] …`
  prints the resolved directory.
- Tests: offline unit tests plus a local-HTTP-server integration test (including
  an LFS-style 302 redirect) covering download, layout, `refs`, subfolders, and
  offline reuse. Design: `20_MODEL_HUB_FETCH.md`.

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
  the typed head (no RoPE, ReLU MLP), marker gather/clamp, scorer (`d→d` then
  `d→1` with GELU), top-1/top-2/entropy pooling, and the action head. Both a
  host `forward_encoder_reference` / `forward_reference` and a tenferro
  `forward_encoder_tenferro` / `forward_tenferro` are provided, with synthetic
  parity tests. The forward uses the exact erf-based GELU: `erf` / `gelu_erf`
  come from the self-hosted `tenferro-ext` extension op and the host reference
  uses the `mlx_erf` port.
- `checkpoint`: loads `encoder/config.json` + `rl_agent_config.json` +
  `model.safetensors` into `LayaWeights` (name sanitizing, config-driven bias
  handling, strict consumption) via the shared `safetensors-io` reader.
- `tokenizer`: a concrete Hugging Face `tokenizer.json` encoder
  (`BpeTokenizer`) implementing the `prompt::Tokenizer` trait — added-token
  extraction, NFC/NFD/NFKC/NFKD + Lowercase/Replace/Prepend normalizers,
  ByteLevel (GPT-2 regex, hand-written scanner) / Metaspace / Whitespace
  pre-tokenizers, and BPE / WordLevel models with byte fallback and `fuse_unk`;
  validated against Julia-generated goldens.
- `agent`: `LayaEngine`, the `DecisionEngine` impl — text/JSON state →
  tokenizer + `build_sequence` markers → forward → calibration → typed answers,
  plus the action probability (`decide`). `load(dir)` wires checkpoint +
  tokenizer + calibration.
- Real-checkpoint parity: `fixtures/laya-tiny/` (a seeded
  `Laya.write_tiny_checkpoint`, regenerated by `tools/gen_laya_fixture.jl`) is
  loaded by the Rust loader; the tokenizer ids and the `forward_reference`
  logits/action match the Julia `DecisionModel` to ~2e-9, and `LayaEngine::load`
  + `system_one` run on the same checkpoint.
- Production parity: `hf-fetch laya` resolved `convaiinnovations/laya@main`
  (commit `7b928d82…`, ~843 MB); with the reference from
  `tools/gen_laya_real_reference.jl` (pinned to that commit) the real tokenizer
  ids match and `forward_reference` matches the Julia `DecisionModel` to
  `1.7e-6` (logits; action within `~1.7e-7` relative). `tests/real_checkpoint.rs`
  skips when the snapshot or reference is absent.
- Tenferro-native execution: `forward_tenferro_cached` threads a shared
  `TensorCache` (from `tenferro-infer`) so weights are not rebuilt per call;
  `LayaEngine` runs this cached tenferro path (160 ms vs 216 ms host at L8B1).
  `tests/model.rs` checks cache-reuse parity.
- Remaining: production `system_one`/`predict` answer parity (prompt rendering +
  calibration end to end). See `23_TENFERRO_NATIVE.md`.

### Phase 5 — `jeff-infer`

- `config`: supported Qwen3.5 `TextConfig` subset and `DecisionConfig`.
- `readout`: temperature-scaled softmax and the choice/noul/score formulas.
- `model`: the Qwen3.5 layer stack — embedding, input/post RMSNorm, full
  attention (GQA-expanded, split `q`/gate, Q/K RMSNorm, partial RoPE, causal
  mask, output projection) or Gated DeltaNet (`tenferro-gated-delta`), the
  SiLU-gated MLP, the final last-position RMSNorm, and the readout. Both a host
  `forward_reference` and a tenferro `forward_tenferro` are provided. The host
  forward runs DeltaNet through the fused recurrent kernel; the tenferro forward
  dispatches each DeltaNet layer through the plan-based `gated_delta` entry.
  `_with` variants take a reusable `GatedDeltaWorkspace`; the engine reuses one
  across rows.
- `checkpoint`: loads `config.json` + `decision_config.json` +
  `model.safetensors` + `readout.safetensors` into `JeffWeights` via the shared
  `safetensors-io` reader — interleaved per-head `[query, gate]` split of the
  fused `q_proj`, consecutive GQA expansion, `a_decay = -exp(A_log)`, and the
  `(channels, 1, taps)` convolution layout.
- `engine`: `JeffEngine`, the prepared-token `DecisionEngine` — leading-padding
  trim + per-row `forward_reference_with` (shared workspace) → readout → typed
  answers, with row `i` answering question `i`.
- Real-fixture parity: against the `extern/JeffClient.jl` synthetic Qwen3.5
  fixture (lengths 1/3/63/64/65, batch 2, left padding), `forward_reference`
  matches the independent PyTorch logits to below `1e-4` (observed worst
  `5.4e-7`) and `forward_tenferro` to below `1e-3` (observed worst `3.0e-7`).
  The tests skip when the fixture submodule is absent.
- Production parity: `hf-fetch jeff` resolved `mstrasser/Jeff-Qwen3.5-0.8B`
  (commit `0f212b3e…`, ~1.7 GB, 0.8B hybrid Qwen3.5); with the reference from
  `tools/gen_jeff_real_reference.jl`, `forward_reference` matches the Julia
  `NativeBackend` logits to `1.8e-5` (scale ~10). `tests/real_checkpoint.rs` is
  `#[ignore]`d (it runs the full 0.8B forward):
  `cargo test --release -p jeff-infer --test real_checkpoint -- --ignored`.
- Backend choice: `JeffEngine` defaults to `JeffBackend::Host` (the fused
  CPU-competitive forward) and offers `JeffBackend::Tenferro`
  (`forward_tenferro_cached` + cached weights) for backend portability; the
  tenferro path is ~10× slower on CPU today (`23_TENFERRO_NATIVE.md`).
- Remaining: natural-language tokenizer.

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
- Fused host recurrent kernel (`recurrent`): projections, causal convolution,
  Q/K normalization, gates, the recurrent scan, output RMSNorm/gate, and output
  projection in one host pass over reusable [`GatedDeltaWorkspace`] buffers —
  no tenferro round-trips and no per-head intermediate allocations.
- Prepared plans/workspaces (`plan`, `workspace`): `AlgorithmChoice`
  (`Auto`/`Reference`/`Recurrent`/`Chunked`), deterministic `resolve_algorithm`
  from config + `BackendCaps` (never input-dependent), `GatedDeltaPlan`, and
  `GatedDeltaWorkspace`. The `gated_delta` direct entry dispatches on the plan.
- `GatedDelta` extension op (`extension`): `ExtensionOp` +
  `define_extension_runtime!` (CPU session route), descriptor fields for the
  layer config, `x`/`mask`/nine weight tensors as inputs, and an
  `EagerSessionGatedDeltaExt` eager helper. The op converts column-major tensor
  inputs to the row-major host layout and runs the fused recurrent kernel.
- Cross-checks: the fused recurrent kernel matches the reference to <1e-6; the
  direct entry matches for every algorithm; the extension op matches the
  reference through the eager session.
- Remaining: CUDA kernels (`cuda` feature; no hardware here to validate) and
  wiring `jeff-infer`'s forward onto the plan/extension path.

## Blocked / needs external input

| Area | Blocker |
|---|---|
| Laya numerical parity (Phase 2) | Tokenizer and forward (encoder + decision head) match `extern/Laya.jl` on the production `convaiinnovations/laya` checkpoint (logits `1.7e-6`); Jeff's forward matches `extern/JeffClient.jl` on `mstrasser/Jeff-Qwen3.5-0.8B` (`1.8e-5`). Full `system_one`/`predict` answer parity (prompt rendering + calibration) is not yet cross-checked. |
| Phase 4 / 7 CUDA | No CUDA hardware here; code can be written but not validated. |
| Phase 8 FP16/BF16 | Deferred by decision: tenferro's public dtype set lacks `F16`/`BF16` at the pinned revision. |
| Phase 10 Apple GPU | tenferro's WebGPU surface is effectively `dot_general` (F32/C32) plus transpose; most primitives are missing. |

## Verification snapshot

- `cargo test --workspace`: decision-core 20, reference-data 4, tenferro-infer 13,
  tenferro-ext 2, cpu-kernels 5, hf-fetch 9, laya-infer 46, jeff-infer 33,
  tenferro-gated-delta 19, jev-client 49 (58 with `--features http`),
  bench-suite 1 (201 total). Laya's `real_checkpoint.rs` adds ~45 s when the
  production snapshot is cached (skips otherwise); Jeff's is `#[ignore]`d.
- `cargo clippy --workspace --all-targets -- -D warnings`: clean.
