# Implementation Roadmap

## Phase 0 — Workspace and baseline

Deliver:

- Cargo workspace
- `decision-core`
- tenferro dependency wiring
- reference fixture format
- CI
- benchmark harness skeleton

Exit criteria:

- basic tenferro CPU operations run
- reference data loader works
- benchmark metadata is captured

---

# Phase 1 — `tenferro-infer` reference primitives

Implement:

- Linear/prepared matmul helper
- embedding gather
- LayerNorm
- RMSNorm
- GELU
- GeGLU
- SiLU
- gated SiLU
- stable softmax
- masked softmax
- RoPE variants
- reference multi-head attention

Exit criteria:

- independent numerical tests
- CPU benchmark for every major primitive
- no model dependency inside primitive crate

---

# Phase 2 — Laya CPU correctness

Implement:

- safetensors loading
- config parsing
- tokenizer
- prompt/collation
- ModernBERT
- decision head
- calibration
- all question types

Exit criteria:

- tiny deterministic fixture passes
- real checkpoint passes
- selected answers match reference
- logits/action within documented tolerance

---

# Phase 3 — Laya CPU optimization

Optimize:

- weight packing
- workspace reuse
- SIMD LayerNorm
- fused residual+norm
- optimized softmax
- optimized attention
- GeGLU
- small decision-head hot spots if necessary
- thread policy

Exit criteria:

- reproducible performance report
- no avoidable large forward allocation
- faster than unoptimized Rust reference
- correctness unchanged

This is the first production-quality milestone.

---

# Phase 4 — Laya CUDA

Implement:

- device-resident weights
- GPU workspace
- compact input upload
- CUDA projections
- fused or optimized attention
- sliding-window attention optimization
- device-resident decision head
- compact result download

Exit criteria:

- real checkpoint parity
- no silent CPU fallback
- no intermediate host download
- GPU synchronization only at intended boundaries
- Laya CUDA benchmark report

---

# Phase 5 — Jeff CPU reference

Implement:

- checkpoint reader
- supported Qwen3.5 config
- embeddings
- RMSNorm
- partial RoPE
- full attention
- reference Gated DeltaNet
- SiLU MLP
- final norm
- Jeff readout

Exit criteria:

- prepared real input cases pass
- layer-by-layer reference validation
- Float32 output within tolerance

---

# Phase 6 — `tenferro-gated-delta` optimization

Implement and benchmark:

- recurrent CPU path
- chunked fallback/reference path
- depthwise convolution optimization
- workspace reuse
- deterministic algorithm selection

Exit criteria:

- faster than reference implementation
- stable across boundary lengths
- numerical parity preserved

---

# Phase 7 — Jeff CUDA

Implement:

- CUDA full attention
- CUDA partial RoPE/grouped-KV preparation
- CUDA causal convolution
- specialized Gated Delta kernel
- device-resident Jeff readout

Exit criteria:

- real checkpoint parity
- no hidden CPU fallback
- reproducible GPU benchmark
- numerical stability across regression suite

---

# Phase 8 — Reduced precision

Add:

- FP16
- BF16

Potential follow-up:

- weight-only INT8

Exit criteria per format:

- documented tolerance
- semantic decision regression
- benchmark improvement
- memory report

---

# Phase 9 — Jev Rust client

Implement independently:

- credentials
- HTTP/TLS
- timeouts
- retry
- resource limits
- typed answers/errors
- mock transport

This phase is independent of tenferro performance work and may be developed in parallel.

---

# Phase 10 — Apple GPU

Before claiming support:

1. enumerate all required operations
2. implement missing WebGPU/Metal coverage
3. pass correctness suite
4. ensure device-resident forward
5. profile
6. add fusion and pooling

Apple GPU is not a v1.0 blocker unless project priorities change.

---

# Release proposal

## v0.1

- optimized Laya CPU

## v0.2

- Laya CUDA

## v0.3

- Jeff CPU

## v0.4

- Jeff CUDA

## v0.5

- FP16/BF16 improvements
- optional Jev client completion

## v1.0

Required:

- Laya CPU + CUDA optimized
- Jeff CPU + CUDA optimized
- real checkpoint tests
- benchmark suite
- stable inference context/workspace model
- documented tolerances
- no silent device fallback
- license/security documentation
