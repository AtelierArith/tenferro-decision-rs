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
- Median of 10 iterations after 3 warm-ups, taken as the best of two rounds run
  in opposite order from a cooled machine. The fanless M4 Air throttles under
  sustained load, so absolute milliseconds drift between sessions; ratios are
  more stable than the times.
- Same prepared inputs on both sides; model loading is measured separately.
- Both Julia numbers below use Apple **Accelerate** BLAS on this Apple-silicon
  host. **Jeff** gets it automatically: `QwenDecisionCore` pulls
  `AppleAccelerate` in as a hard dependency, so `cpu_settings().accelerate ==
  true` and the forward uses the chunked DeltaNet (`delta_chunk_size = 64`).
  **Laya**'s default `CPUBackend` is OpenBLAS, so its number is the explicit
  `AccelerateBackend` fast path (`tools/bench_laya_real_accelerate.jl`).

### Jeff

Checkpoint `mstrasser/Jeff-Qwen3.5-0.8B`. Rust "oracle" is the correctness
reference (`forward_reference`); `host_opt` is the default optimized host
forward.

| length | Julia (Accelerate) | Rust oracle | Rust host_opt | host_opt / Julia |
|---:|---:|---:|---:|---:|
| L8 | 67.0 ms | 60.9 ms | 63.7 ms | 0.95× |
| L16 | 70.2 ms | 85.9 ms | 85.9 ms | 1.22× |
| L64 | 126.2 ms | 282.5 ms | 227.6 ms | 1.80× |

Rust host vs the tenferro-native forward at L8 (the tenferro path is timed only
at the shortest sequence; `HostRecurrent` is the default fused recurrent
DeltaNet extension op, `TensorNative` is tensor-only):

| path | ms | vs host_opt | vs Julia (Accel.) |
|---|---:|---:|---:|
| Rust host_opt | 63.7 | 1.00× | 0.95× |
| Rust tenferro `HostRecurrent` (cached) | 93.0 | 1.46× | 1.39× |
| Rust tenferro `TensorNative` (cached) | 120.3 | 1.89× | 1.80× |
| Rust tenferro `HostRecurrent` (fresh cache) | 1568.8 | 24.6× | 23.4× |

### Laya

Checkpoint `convaiinnovations/laya` (`1c5edc17`).

| shape | Julia (Accelerate) | Rust host | host / Julia |
|---|---:|---:|---:|
| L8 B1 | 51.9 ms | 44.1 ms | 0.85× |
| L64 B1 | 83.0 ms | 213.5 ms | 2.57× |
| L8 B8 | 80.9 ms | 165.1 ms | 2.04× |

Rust host vs the tenferro-native forward at L8 B1:

| path | ms | vs host | vs Julia (Accel.) |
|---|---:|---:|---:|
| Rust host | 44.1 | 1.00× | 0.85× |
| Rust tenferro (cached) | 61.0 | 1.38× | 1.18× |
| Rust tenferro (fresh cache) | 249.8 | 5.66× | 4.81× |

### Model load

| model | Julia | Rust |
|---|---:|---:|
| Jeff | 2364 ms | 2619 ms |
| Laya | 407 ms | 1224 ms |

### Reading these numbers

- On this Apple-silicon host, Julia's Accelerate-backed GEMMs (AMX) dominate
  **Laya**: at L64 B1, 83 ms against Rust's 213 ms. Rust's row-blocked
  `matrixmultiply` wins only the tiny L8 B1 decode (44 vs 52 ms).
- **Jeff is close at short lengths** (Rust `host_opt` is at parity at L8,
  0.95×) but the gap grows to 1.80× at L64, where Julia's chunked DeltaNet and
  Accelerate GEMMs pull ahead of Rust's recurrent scan.
- The **tenferro-native forward is 1.4–1.9× behind the Rust host path** with a
  reused tensor cache; without caching it rebuilds every weight tensor per call
  and is 4.8× (Laya) / 23.4× (Jeff) slower. `HostRecurrent` is faster than
  `TensorNative`, so the `GatedDelta` extension op and `JeffEngine` default to
  the host kernels (`JeffBackend::Auto`).
- These are Apple-silicon numbers. The x86_64 picture is different (Julia's Laya
  lead is smaller; Rust `host_opt` was at parity or ahead at L8/L16 for Jeff):
  see
  [`docs/agents/specs/docs/21_SPEED_COMPARISON.md`](docs/agents/specs/docs/21_SPEED_COMPARISON.md).

Reproduce:

```sh
tools/bench_compare.sh --jeff <JEFF_CKPT_DIR> --laya <LAYA_CKPT_DIR> \
    --threads 8 --warmup 3 --iters 10 --json /tmp/bench

# Optional Apple-silicon Laya Accelerate run:
ACC=$(mktemp -d)
julia --project="$ACC" -e 'using Pkg; Pkg.develop(path="extern/Laya.jl"); Pkg.add("AppleAccelerate")'
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
