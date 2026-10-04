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
| `tenferro-gated-delta` | Gated DeltaNet crate for Jeff (host reference + tenferro chunked scan + full layer) |
| `safetensors-io` | Shared dependency-light safetensors reader for the checkpoint loaders |
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
the ModernBERT + decision-head forward, and a text/JSON engine. Validating Laya
against real checkpoint assets, exact erf GELU, and plans/workspaces remain.
See
[`docs/agents/specs/docs/18_IMPLEMENTATION_STATUS.md`](docs/agents/specs/docs/18_IMPLEMENTATION_STATUS.md)
for the live status and blockers (CUDA hardware, tenferro `erf`/F16/BF16,
WebGPU coverage).

## Building and testing

```sh
cargo test --workspace
cargo test -p decision-core --features serde
cargo clippy --workspace --all-targets -- -D warnings
cargo bench -p bench-suite
```

The MVP targets the tenferro CPU provider (`cpu-faer`); CUDA follows in later
phases.
