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
| `tenferro-ext` | Self-hosted tenferro extension ops (`erf`, exact GELU) used by Laya |
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
[`tools/bench_compare.sh`](tools/bench_compare.sh). Measured on an Apple M4
(10 cores, macOS 26, Julia 1.13.1, rustc 1.99.0, release build):

- 8 threads on both sides: Julia `-t 8`, Rust `RAYON_NUM_THREADS=8`.
- Median of 10 iterations after 3 warm-ups, from one cooled run per
  configuration. The fanless M4 Air throttles under sustained load, so absolute
  milliseconds drift between sessions; ratios are more stable than the times.
- Same prepared inputs on both sides; model loading is measured separately.
- Two Julia configurations are shown. **OpenBLAS** is the default portable
  policy (BLAS=1, Julia-parallel projections; `blas_threads=1`). **Accelerate**
  is the opt-in fast path: `using AppleAccelerate` forwards BLAS to Accelerate
  and `QwenDecisionCore`'s extension switches to the chunked DeltaNet
  (`delta_chunk_size = 64`, `blas_threads=8`). `AppleAccelerate` is a weak
  dependency, so it must be added to the environment
  (`tools/bench_jeff_real_accelerate.jl` / `tools/bench_laya_real_accelerate.jl`
  and `bench_compare.sh --acc-env`).

### Jeff

Checkpoint `mstrasser/Jeff-Qwen3.5-0.8B`. Rust "oracle" is the correctness
reference (`forward_reference`); `host_opt` is the default optimized host
forward.

| length | Julia (OpenBLAS) | Julia (Accelerate) | Rust oracle | Rust host_opt | host_opt / Julia (best) |
|---:|---:|---:|---:|---:|---:|
| L8 | 72.2 ms | 67.7 ms | 64.0 ms | 63.7 ms | 0.94× |
| L16 | 96.0 ms | 74.4 ms | 86.1 ms | 85.6 ms | 1.15× |
| L64 | 231.5 ms | 127.7 ms | 280.9 ms | 222.9 ms | 1.75× |

Rust host vs the tenferro-native forward at L8 (the tenferro path is timed only
at the shortest sequence; `HostRecurrent` is the default fused recurrent
DeltaNet extension op, `TensorNative` is tensor-only):

| path | ms | vs host_opt | vs Julia (Accel.) |
|---|---:|---:|---:|
| Rust host_opt | 63.7 | 1.00× | 0.94× |
| Rust tenferro `HostRecurrent` (cached) | 96.1 | 1.51× | 1.42× |
| Rust tenferro `TensorNative` (cached) | 131.8 | 2.07× | 1.95× |
| Rust tenferro `HostRecurrent` (fresh cache) | 1611.4 | 25.3× | 23.8× |

### Laya

Checkpoint `convaiinnovations/laya` (`1c5edc17`).

| shape | Julia (OpenBLAS) | Julia (Accelerate) | Rust host | host / Julia (best) |
|---|---:|---:|---:|---:|
| L8 B1 | 71.0 ms | 53.5 ms | 48.0 ms | 0.90× |
| L64 B1 | 228.2 ms | 83.8 ms | 218.3 ms | 2.61× |
| L8 B8 | 228.5 ms | 82.1 ms | 174.2 ms | 2.12× |

Rust host vs the tenferro-native forward at L8 B1:

| path | ms | vs host | vs Julia (Accel.) |
|---|---:|---:|---:|
| Rust host | 48.0 | 1.00× | 0.90× |
| Rust tenferro (cached) | 62.1 | 1.29× | 1.16× |
| Rust tenferro (fresh cache) | 260.1 | 5.42× | 4.86× |

### Model load

| model | Julia | Rust |
|---|---:|---:|
| Jeff | 2562 ms | 2498 ms |
| Laya | 427 ms | 1414 ms |

### Reading these numbers

- On this Apple-silicon host, Julia's Accelerate-backed GEMMs (AMX) dominate at
  longer shapes: Jeff L64 **231 → 128 ms** and Laya L64 **228 → 84 ms** versus
  OpenBLAS. The short L8 decode is nearly identical on both BLAS backends.
- **Rust `host_opt` wins the tiny L8 decode** for both models (0.94× Jeff, 0.90×
  Laya) but is left behind at L16/L64, where Julia's Accelerate GEMMs and
  chunked DeltaNet pull ahead (Jeff 1.75× at L64; Laya 2.6× at L64).
- The **tenferro-native forward is 1.3–2.1× behind the Rust host path** with a
  reused tensor cache; without caching it rebuilds every weight tensor per call
  and is 4.9× (Laya) / 23.8× (Jeff) slower. `HostRecurrent` is faster than
  `TensorNative`, so the `GatedDelta` extension op and `JeffEngine` default to
  the host kernels (`JeffBackend::Auto`).
- These are Apple-silicon numbers with an opt-in Accelerate. The x86_64 picture
  (Julia's default BLAS there is OpenBLAS) is in
  [`docs/agents/specs/docs/21_SPEED_COMPARISON.md`](docs/agents/specs/docs/21_SPEED_COMPARISON.md).

Reproduce:

```sh
tools/bench_compare.sh --jeff <JEFF_CKPT_DIR> --laya <LAYA_CKPT_DIR> \
    --threads 8 --warmup 3 --iters 10 --json /tmp/bench

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
