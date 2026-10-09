# tenferro-decision-rs

A Rust typed-decision inference stack built on
[tenferro-rs](https://github.com/tensor4all/tenferro-rs): **Laya** first, then
**Jeff**, then the remote **Jev** client.

The design package lives in [`docs/agents/specs/`](docs/agents/specs/README.md).
Read `06_ROADMAP.md` for the phase plan and `01_DESIGN.md` for the crate
architecture.

## Workspace

| crate | role |
|---|---|
| `decision-core` | Backend-independent questions/answers, `Content`, and the `DecisionEngine` seam |
| `tenferro-infer` | Shared inference primitives (norm, activations, softmax, RoPE, attention) |
| `laya-infer` | Laya engine (config, calibration, prompt, tokenizer, checkpoint loading, ModernBERT + decision-head forward, `DecisionEngine`) |
| `jeff-infer` | Jeff engine (config, readout, Qwen3.5 layer stack, checkpoint loading, prepared-token `DecisionEngine`, real-fixture parity) |
| `tenferro-gated-delta` | Gated DeltaNet crate for Jeff (host reference + fused recurrent kernel + tenferro chunked scan + full layer + plans/workspaces + `GatedDelta` extension op) |
| `safetensors-io` | Shared dependency-light safetensors reader for the checkpoint loaders |
| `tenferro-ext` | Self-hosted tenferro extension ops on `cpu-kernels`: exact GELU/`erf`, `linear`/`gemm_bias`, feature-first LayerNorm, GeGLU, gated SiLU, feature-last RMSNorm, and the fused Laya `split + RoPE + attention` and Jeff full-attention blocks |
| `cpu-kernels` | Host CPU kernels: BLAS-class `matrixmultiply` GEMM parallelized with `rayon`, shared by all engines |
| `hf-fetch` | Hugging Face Hub checkpoint fetcher (Julia-compatible cache and env; `hf-fetch` CLI) |
| `jev-client` | Independent TypeSafe System One API client |
| `reference-data` | Reference fixture format and loader (test support) |
| `bench-suite` | Benchmark harness skeleton and metadata capture |

`decision-core` and `jev-client` never depend on tenferro.

## Status

Phase 0 (workspace), Phase 1 (`tenferro-infer` primitives), and Phase 9
(`jev-client`) are implemented and tested. Phases 2/5/6 now have loadable
checkpoints and `DecisionEngine` wiring: `jeff-infer` has safetensors checkpoint
loading, a prepared-token engine, and real-fixture parity against the
`extern/JeffClient.jl` synthetic Qwen3.5 model's independent PyTorch logits,
while `laya-infer` has a concrete `tokenizer.json` encoder, checkpoint loading,
the ModernBERT + decision-head forward (with the exact erf GELU from the
self-hosted `tenferro-ext` extension op), and a text/JSON engine, validated
against a seeded Julia-generated fixture (`fixtures/laya-tiny/`, tokenizer and
forward parity to ~2e-9). The production `convaiinnovations/laya` checkpoint is
fetchable with `hf-fetch` and its tokenizer/forward match `extern/Laya.jl`
(logits `1.7e-6`); Jeff's production checkpoint
(`mstrasser/Jeff-Qwen3.5-0.8B`) can also be fetched, and answer-level parity
(prompt + calibration) is next.
See
[`docs/agents/specs/docs/18_IMPLEMENTATION_STATUS.md`](docs/agents/specs/docs/18_IMPLEMENTATION_STATUS.md)
for the live status and blockers (CUDA hardware, tenferro `erf`/F16/BF16,
WebGPU coverage).

## Usage

Fetch a checkpoint with the `hf-fetch` CLI; it prints the local snapshot
directory that the engines load from:

```sh
cargo run -p hf-fetch -- laya   # -> ~/.cache/huggingface/.../snapshots/<sha>
cargo run -p hf-fetch -- jeff
```

Both engines implement `decision_core::DecisionEngine` and answer a
`QuestionSet` against a `State`. The snippets below are also runnable examples:

```sh
cargo run --release -p laya-infer --example laya_system_one -- <CHECKPOINT_DIR> [TEXT]
cargo run --release -p jeff-infer --example jeff_system_one -- <CHECKPOINT_DIR> [--tenferro]
```

### Laya — natural language

Laya is text/JSON only; `LayaEngine::load` reads the encoder/agent configs,
`model.safetensors`, the `tokenizer/` assets, and the calibration in one call.

```rust
use decision_core::{
    ChoiceQuestion, Content, DecisionEngine, NoulCriteria, NoulQuestion, Question, QuestionSet,
    ScoreQuestion, State,
};
use laya_infer::agent::LayaEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Directory printed by `hf-fetch laya`.
    let mut engine = LayaEngine::load("/path/to/laya/snapshot")?;

    let mut questions = QuestionSet::new();
    questions.push(
        "next_action",
        Question::Choice(ChoiceQuestion::new(
            Content::string("What should the character do next?"),
            vec![
                ("greet".into(), Content::string("greet the visitor")),
                ("wait".into(), Content::string("keep waiting")),
                ("leave".into(), Content::string("walk away")),
            ],
        )?),
    )?;
    questions.push(
        "risk",
        Question::Score(ScoreQuestion::new(
            Content::string("How risky is this plan?"),
            vec!["low".into(), "medium".into(), "high".into()],
        )?),
    )?;
    questions.push(
        "agrees",
        Question::Noul(NoulQuestion::new(
            Content::string("Does the character agree?"),
            NoulCriteria {
                truthy: Some(Content::string("yes")),
                falsy: Some(Content::string("no")),
            },
        )?),
    )?;

    let state = State::Text("A traveler knocks on the door.".to_string());
    let answers = engine.system_one(&state, &questions)?;
    for ((id, _), answer) in questions.questions().iter().zip(&answers) {
        println!("{id}: {answer:?}");
    }
    Ok(())
}
```

`State::Json(Content::object([...]))` works the same way. Use
`engine.decide(&state, &questions)?` instead of `system_one` when you also want
each answer's action probability (the action-head output) next to the typed
answer.

### Jeff — prepared tokens

Jeff's `DecisionEngine` currently accepts `PreparedState` only: one token row per
question, `row i` answering question `i`, with leading padding trimmed by the
engine. The natural-language tokenizer is not wired yet.

```rust
use decision_core::{
    ChoiceQuestion, Content, DecisionEngine, PreparedState, Question, QuestionSet, State,
};
use jeff_infer::checkpoint::load_checkpoint;
use jeff_infer::engine::{JeffBackend, JeffEngine};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Directory printed by `hf-fetch jeff`.
    let checkpoint = load_checkpoint("/path/to/jeff/snapshot")?;
    let mut engine = JeffEngine::new(checkpoint.config, checkpoint.decision, checkpoint.weights)?;

    // One prepared row per question; `row 0` answers `questions[0]`.
    let state = State::Prepared(PreparedState {
        input_ids: vec![vec![1, 2, 3, 4]],
        attention_mask: vec![vec![true; 4]],
    });

    let mut questions = QuestionSet::new();
    questions.push(
        "next_action",
        Question::Choice(ChoiceQuestion::new(
            Content::string("Pick the next action."),
            vec![
                ("a".into(), Content::string("first")),
                ("b".into(), Content::string("second")),
                ("c".into(), Content::string("third")),
            ],
        )?),
    )?;

    for answer in engine.system_one(&state, &questions)? {
        println!("{answer:?}");
    }
    Ok(())
}
```

`JeffEngine::new` defaults to `JeffBackend::Auto` (the optimized host forward);
`JeffEngine::with_backend(config, decision, weights, JeffBackend::Tenferro)`
selects the tenferro-native forward for backend portability, and
`engine.logits(&prepared)?` exposes the raw per-row readout logits behind the
typed answers.

## Benchmarks

CPU comparison against the Julia reference implementations
(`extern/JeffClient.jl` and `extern/Laya.jl`), driven by
[`tools/bench_compare.sh`](tools/bench_compare.sh). Measured on an Apple
M2 Max (12 cores, macOS 26, Julia 1.13.1, rustc 1.98.1, release build):

- 8 threads on both sides: Julia `-t 8`, Rust `RAYON_NUM_THREADS=8`.
- Median of 30 iterations after 5 warm-ups. Absolute milliseconds drift
  between sessions; ratios are more stable than the times.
- Same prepared inputs on both sides; model loading is measured separately.
- Two Julia configurations are shown. **OpenBLAS** is the default portable
  policy (Jeff: `BLAS=1`, Julia-parallel projections). **Accelerate** is the
  opt-in fast path: `using AppleAccelerate` forwards BLAS to Accelerate (AMX).
  `AppleAccelerate` is a weak dependency, so it must be added to the
  environment (`tools/bench_*_real_accelerate.jl` and
  `bench_compare.sh --acc-env`).
- On Apple Silicon the default **Jeff** env is OpenBLAS + `BLAS=1`. A *stale*
  `QwenDecisionCore` revision used to force-load Accelerate here, so earlier
  Jeff "OpenBLAS" rows were really Accelerate/`BLAS=8`; update
  `QwenDecisionCore` (`Pkg.update`) to get the audited default.

### Optional oneDNN CPU provider

Build Laya with `--features onednn` to use the owned oneDNN F32 projection,
GeGLU, and selected LayerNorm extension operations. Install the oneDNN headers
and shared library in the build environment; for a nonstandard installation set
`ONEDNN_INCLUDE_DIR` and `ONEDNN_LIB_DIR`, and make the shared library available
to the runtime loader. This feature requires a C++17 compiler. The default build
does not require oneDNN.

For the measured eight-thread configuration, set `OMP_NUM_THREADS=8`,
`RAYON_NUM_THREADS=8`, and `OMP_WAIT_POLICY=PASSIVE`. Packed weights are owned
by the model's preparation cache; activation and aligned oneDNN scratch buffers
are reused by the session. Primitives use user-provided scratch so sequential
calls can move between threads. Arithmetic uses strict F32 library operations.
Other backends continue to use the tenferro composition.

Current evidence and raw samples are in
[`fixtures/bench-onednn-owned-2026-10-09`](fixtures/bench-onednn-owned-2026-10-09).
Two Rust runs beat measured original Python medians at all four Laya shapes;
Jeff also wins at its three measured shapes. These results apply to the optional
Laya feature and the default Jeff HostOpt path, and do not establish performance
for other backends. See `docs/agents/specs/docs/21_SPEED_COMPARISON.md` for
conditions, timing drift and correctness evidence.

### Jeff

Checkpoint `mstrasser/Jeff-Qwen3.5-0.8B`. `Rust (opt)` is `host_opt`, the
default optimized host forward; `Rust (tenferro-rs)` is the cached
tenferro-native forward (`--example bench_jeff_tenferro_gap`, all shapes):

| length | Julia (OpenBLAS) | Julia (Accelerate) | Rust (opt) | Rust (tenferro-rs) |
|---:|---:|---:|---:|---:|
| L8 | 68.9 ms | 114.7 ms | 70.6 ms | 78.2 ms |
| L16 | 90.9 ms | 112.1 ms | 90.2 ms | 98.1 ms |
| L64 | 216.6 ms | 134.9 ms | 208.6 ms | 212.0 ms |

Tuning variants: the tensor-only `TensorNative` DeltaNet is 126.5 ms at L8
(1.79× `host_opt`); `forward_tenferro` without the tensor cache rebuilds every
weight per call (553 ms at L8, ~7.8×).

### Laya

Checkpoint `convaiinnovations/laya` (`1c5edc17`). `Rust (opt)` is the optimized
host forward (`forward_reference`); `Rust (tenferro-rs)` is the cached
tenferro-native forward (`--example bench_laya_tenferro_gap`, all shapes):

| shape | Julia (OpenBLAS) | Julia (Accelerate) | Rust (opt) | Rust (tenferro-rs) |
|---|---:|---:|---:|---:|
| L8 B1 | 90.7 ms | 88.4 ms | 45.2 ms | 49.8 ms |
| L16 B1 | 120 ms | 91.5 ms | 67.6 ms | 69.6 ms |
| L64 B1 | 286.3 ms | 88.5 ms | 249.3 ms | 169.0 ms |
| L8 B8 | 250.8 ms | 85.9 ms | 173.2 ms | 158.3 ms |

`forward_tenferro` without the tensor cache rebuilds every weight per call
(272.9 ms at L8 B1, ~6×).

### Model load

| model | Julia | Rust |
|---|---:|---:|
| Jeff | 2836 ms | 3268 ms |
| Laya | 490 ms | 1522 ms |

### Reading these numbers

- On this Apple-silicon host Julia's best BLAS depends on the shape: OpenBLAS
  (`BLAS=1`, Julia-parallel projections) wins the decode rows (Jeff L8/L16,
  Laya L8–L16), while Accelerate (AMX) dominates longer shapes (Jeff L64
  **216.6 → 134.9 ms**, Laya L64 **286.3 → 88.5 ms**).
- **Correction:** a stale `QwenDecisionCore` revision forced `AppleAccelerate`
  on Apple Silicon even in the default env, so the earlier Jeff "OpenBLAS" rows
  were Accelerate/`BLAS=8` and inflated Julia. With the current
  `QwenDecisionCore` main the default Jeff env is OpenBLAS + `BLAS=1`; the Jeff
  L8 Accelerate row (114.7 ms) is ~1.7× the OpenBLAS row (68.9 ms), and Julia is
  at parity with Rust `host_opt` there (1.02×).
- At decode Rust is competitive with or ahead of Julia (Jeff L8/L16 ≈ 1.0×,
  Laya L8 B1 `host_opt` 0.51× the best Julia); at L64 Julia's Accelerate GEMMs
  pull ahead (Laya 2.82×, Jeff 1.55×).
- The **tenferro-native forward now reaches the Rust host path**, via
  self-hosted `cpu-kernels` extension ops (Laya: `linear`/`gemm_bias`,
  feature-first LayerNorm, GeGLU, and the fused split + RoPE + attention block;
  Jeff: `linear`, full attention, feature-last RMSNorm, gated SiLU). Cached
  tenferro is **0.68–1.08× the Rust host** for Laya and **1.02–1.12×
  `host_opt`** for Jeff; without the tensor cache it rebuilds every weight per
  call and is ~3.1× (Laya) / 7.8× (Jeff) slower.
- These are Apple-silicon numbers. The x86_64 picture (Julia's default BLAS
  there is OpenBLAS) is in
  [`docs/agents/specs/docs/21_SPEED_COMPARISON.md`](docs/agents/specs/docs/21_SPEED_COMPARISON.md).

Reproduce:

```sh
tools/bench_compare.sh --jeff <JEFF_CKPT_DIR> --laya <LAYA_CKPT_DIR> \
    --threads 8 --warmup 5 --iters 30 --json /tmp/bench

# Add the Apple-silicon Accelerate runs (AppleAccelerate is now a weak
# dependency, so opt in explicitly):
ACC=$(mktemp -d)
julia --project="$ACC" -e 'using Pkg;
    Pkg.develop(path="extern/JeffClient.jl"); Pkg.develop(path="extern/Laya.jl");
    Pkg.add(["AppleAccelerate", "JSON"])'
tools/bench_compare.sh --jeff <JEFF_CKPT_DIR> --laya <LAYA_CKPT_DIR> --acc-env "$ACC"
```

## Building and testing

```sh
cargo test --workspace
cargo test -p decision-core --features serde
cargo clippy --workspace --all-targets -- -D warnings
cargo bench -p bench-suite
```

The MVP targets the tenferro CPU provider (`cpu-faer`); CUDA follows in later
phases.
