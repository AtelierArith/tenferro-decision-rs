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
- Production answer parity (2026-10-08): `tests/production_answers.rs` compares
  the upstream bundled email state/questions against Julia `prepare`, `collate`,
  `DecisionModel`, and `predict`. Token ids and markers match exactly; batched
  logits and action-head outputs match within floating-point tolerances;
  choice/score/noul answers, calibration, and action probabilities match to
  `2e-4` (the Julia answers are rounded to four decimals). The captured action
  probabilities are saturated at 1, so the test also compares the raw action
  logits to verify the trained head. The test skips when the pinned checkpoint
  or `fixtures/laya-real/answers.json` capture is absent; regenerate with
  `tools/gen_laya_answers.jl`.
- `LayaEngine` now collates questions into padded tenferro batches with marker
  masks, bounded to 16 rows per forward by default. `with_batch_size` controls
  the limit; synthetic regression tests compare batched and chunked execution
  against individual rows. Prepared-token states remain unsupported.

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
  The same production reference also records Julia `decide` answers for
  choice/noul/score, including an interior mask hole and left padding.
  `engine_answers_match_production_julia_decide` verifies the production
  temperature, chosen labels, probabilities, confidence, score, and legend
  through `JeffEngine::system_one` (absolute answer tolerance `2e-4`). Both
  ignored acceptance tests were run successfully on 2026-10-08. Production
  Julia/Rust timings and the original Python comparison are recorded in
  `21_SPEED_COMPARISON.md`.
- Backend choice: `JeffEngine` defaults to `JeffBackend::Host` (the fused
  CPU-competitive forward) and offers `JeffBackend::Tenferro` for backend
  portability. The tenferro path uses `forward_tenferro_cached` plus the
  fully tensor-native, head-batched `tensor_layer::delta_layer_tenferro_native`;
  it is now ~1.9× the host path on CPU (L8 342 ms vs 177 ms), down from ~10×
  (`23_TENFERRO_NATIVE.md`).
- Natural-language input (2026-10-08): `JeffTokenizer` loads local
  `tokenizer.json` with Qwen's NFC + regex/byte-level BPE, answer codes from
  `decision_config.json`, and the compiled checkpoint chat template. The
  state-first prompt renders Text/Json, choice/score/noul, and ordered Python
  JSON spelling; thinking is disabled, matching original Jeff. No tokenizer
  networking features are enabled, and overlong inputs fail without truncation.
- `JeffEngine::load` attaches tokenizer assets when present. Raw-weight
  constructors remain prepared-only until `with_tokenizer`; `prepare` exposes
  the exact prepared rows. `jeff_system_one --text STATE` demonstrates the API.
- Goldens in `fixtures/jeff-real/tokenizer.json` compare prompts and token ids
  exactly against original Python Jeff. `extern/JeffClient.jl` accepts prepared
  tokens and has no tokenizer API, so `tools/gen_jeff_text_answers.jl` captures
  its `decide` answers using those independently produced token ids. The ignored
  production Text/Json acceptance test then compares Rust answers with Julia.
  Regenerate first with `tools/gen_jeff_tokenizer_reference.py`, then the Julia
  script; ordinary tests use a small offline WordLevel tokenizer and synthetic
  model weights.

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
- Fully tensor-native layer (`tensor_layer`): tensor-in/tensor-out, **head
  batched** across value heads, so it composes with the tenferro forward with no
  host round-trips. The `GatedDelta` extension op stays CPU-only (the extension
  API exposes no session-op helpers).
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
  Plan resolution rejects declared CUDA and unavailable backends with
  `UnsupportedConfig` rather than selecting a CPU fallback. `resolve_algorithm`
  and `GatedDeltaPlan::resolve` return `Result`; explicit CPU `from_config`
  remains infallible.
- `GatedDelta` extension op (`extension`): `ExtensionOp` +
  `define_extension_runtime!` (CPU session route), descriptor fields for the
  layer config, `x`/`mask`/nine weight tensors as inputs, and an
  `EagerSessionGatedDeltaExt` eager helper. The op converts column-major tensor
  inputs to the row-major host layout and runs the fused recurrent kernel.
- Cross-checks: the fused recurrent kernel matches the reference to <1e-6; the
  direct entry matches for every algorithm; the extension op matches the
  reference through the eager session.
- Optional CUDA kernel component: `cuda::CudaKernels` compiles and retains raw
  handles for causal conv/SiLU, warp-owned register-state recurrent scan with
  fused Q/K normalization and beta/decay gates, and output RMSNorm/gate.
  `tools/check_cuda_kernels.py` validates PTX generation and entry-point ABIs
  with NVRTC without a GPU. This does not enable full CUDA layer dispatch.
- Remaining: CUDA layer integration, large-key chunked kernels and device
  parity (no hardware here to validate), and
  wiring `jeff-infer`'s forward onto the plan/extension path.

## Blocked / needs external input

| Area | Blocker |
|---|---|
| Laya numerical parity (Phase 2) | Tokenizer and forward (encoder + decision head) match `extern/Laya.jl` on the production `convaiinnovations/laya` checkpoint (logits `1.7e-6`); Jeff's forward matches `extern/JeffClient.jl` on `mstrasser/Jeff-Qwen3.5-0.8B` (`1.8e-5`). Production `system_one`/`predict` parity (prompt rendering + calibration) is verified on the bundled Laya email example; Jeff prepared-token answers match Julia `decide`. |
| Phase 4 / 7 CUDA | Implemented and hardware-validated on an RTX 3060 (CUDA 13 devcontainer); see "CUDA engines (2026-10-10)" below. Remaining gaps listed there. |
| Phase 8 FP16/BF16 | Deferred by decision: tenferro's public dtype set lacks `F16`/`BF16` at the pinned revision. |
| Phase 10 Apple GPU | tenferro's WebGPU surface is effectively `dot_general` (F32/C32) plus transpose; most primitives are missing. |

## Verification snapshot

- `cargo test --workspace`: decision-core 20, reference-data 4, tenferro-infer 13,
  tenferro-ext 2, cpu-kernels 5, hf-fetch 9, laya-infer 46, jeff-infer 33,
  tenferro-gated-delta 19, jev-client 49 (58 with `--features http`),
  bench-suite 1 (201 total). Laya's `real_checkpoint.rs` adds ~45 s when the
  production snapshot is cached (skips otherwise); Jeff's is `#[ignore]`d.
- `cargo clippy --workspace --all-targets -- -D warnings`: clean.

## CUDA engines (2026-10-10)

Validated on an RTX 3060 (12 GB, compute 8.6) in `.devcontainer/` (CUDA
13.0.3, cuTENSOR 13), pinned tenferro-rs `471c427`.

- API: `tenferro_ext::Device { Cpu, Cuda(ordinal) }` (re-exported as
  `laya_infer::agent::Device` / `jeff_infer::engine::Device`);
  `LayaEngine::{with_device, load_with_device, device}`;
  `JeffEngine::{with_device, load_with_device, device, delta_kernel}`. A CUDA
  device selects `JeffBackend::Tenferro` with the new `DeltaKernel::Cuda`.
  `jeff_infer::model::forward_tenferro_device` retains the host DeltaNet
  workspace, `CudaDeltaWorkspaces` and the weight cache across calls.
- Features: `laya-infer/cuda`, `jeff-infer/cuda` (forwarding to
  `tenferro-ext/cuda`, `tenferro-gated-delta/cuda`, `tenferro-ad/cuda`).
- Jeff DeltaNet: `CudaRequest::layer_time_first` (time-first activations,
  stacked `(qkv|z|a|b)` projection from `prepare_time_first_weights`, workspaces
  resized when the request length changes) inside one
  `with_cuda_request_cached` scope per forward (module compiled once per
  thread/runtime). The recurrent scan kernel stages Q/K, gates and V in
  shared memory per 16/32-token tile (was one uncoalesced warp per row).
- Fused CUDA kernels (`tenferro_ext::cuda_fused`, `Fusion`): RMSNorm /
  LayerNorm, gated SiLU (stacked gate|up), GeGLU, bias+activation, embedding
  gather, Jeff full attention (norm+RoPE prep, causal gated attention) and
  Laya attention (split+RoPE prep, biased attention).
- Fixed the three failing `cuda_stages` gates: an F64 state literal in the
  decay test, and eager `scale_real` (host scalar import) in the raw layer
  tests, replaced by an explicitly uploaded scalar.
- Device attention uses an additive F32 bias
  (`tenferro_infer::attention::attention_with_bias`): the CUDA provider cannot
  materialize a broadcast Bool mask. Jeff linear weights use `(out, in)` on
  non-CPU backends (the CPU `(in, out)` cache layout was only correct for the
  CPU extension).
- Accuracy (max |Δ|): Jeff production L8 CUDA vs CPU 7.2e-6, vs Julia 7.6e-6;
  parcel (101 active of 256) vs CPU 9.1e-6, vs PyTorch reference 9.5e-6
  (scale 18.6); synthetic stacks vs host oracle ≤ 6e-7. Laya logits vs CPU
  1.1e-6 (B1) / 1.1e-5 (B5 batch), vs Julia 1.7e-6 / 8.6e-6; action relative
  ≤ 2e-6.
- Tests: CPU `cargo test --workspace` 252 passed / 0 failed / 3 ignored;
  with all CUDA features 256 passed; `--ignored` with CUDA features 20 passed
  (CUDA hardware: gated-delta 9, jeff 3, laya 2, tenferro-ext 3; plus the 3
  CPU production-Jeff gates).
- Remaining: warm latency is now bound by tenferro eager per-op host
  overhead plus GEMM time (see `23_TENFERRO_NATIVE.md`); large-key (>256)
  DeltaNet is not supported by the time-first layer; Laya marker pooling still
  round-trips through the host; the first call pays NVRTC compilation and the
  weight upload (7–15 s for Jeff).

## Raw single-stream CUDA forward (2026-10-11)

On a CUDA device `LayaEngine` and `JeffEngine` now default to
`CudaPath::Raw` (`tenferro_ext::CudaPath`, re-exported by both engines;
`with_cuda_path`, `cuda_path()`, `TENFERRO_DECISION_CUDA_PATH=native`): one
`with_raw` scope per forward running cuBLAS and NVRTC kernels on tenferro's
stream (`tenferro_ext::raw_exec`, `laya_infer::cuda_raw`,
`jeff_infer::cuda_raw`, `tenferro_gated_delta::raw`). The tenferro-native
device forward is unchanged and selectable; it is used automatically when the
raw path does not support a model (Laya: head_dim 64 and hidden <= 1024; Jeff:
hidden <= 1024, head_dim a multiple of 32, DeltaNet key_dim <= 256, taps <= 8;
rows longer than 8192 tokens). Design, measurements and the comparison with
the Julia GPU runtimes: `23_TENFERRO_NATIVE.md`.

