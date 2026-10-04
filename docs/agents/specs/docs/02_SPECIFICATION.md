# Normative Specification

Keywords **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative.

## 1. Scope

The implementation MUST provide high-performance inference for:

- Laya typed-decision models
- Jeff text-only Qwen3.5 typed-decision models

The implementation SHOULD provide a Rust Jev/TypeSafe API client, but Jev is independent from local tensor inference.

---

## 2. Primary backend requirements

### CPU

CPU inference MUST be a supported production path.

The CPU path MUST:

- run without Python
- run without Julia
- support real checkpoints
- avoid implicit GPU dependencies
- expose reproducible benchmarks
- reuse steady-state workspaces
- use optimized matrix multiplication
- provide reference-equivalent decisions

### CUDA

CUDA inference MUST be a supported production target before v1.0.

The CUDA path MUST:

- keep model weights device-resident after load
- keep intermediate activations device-resident
- avoid silent CPU fallback
- avoid hidden host/device copies
- synchronize only at explicit boundaries
- return typed errors for unsupported operations
- expose end-to-end latency including output synchronization

### Apple GPU

Apple GPU MAY be implemented after CPU/CUDA.

It MUST NOT be claimed production-ready until required operations and numerical tests pass on real hardware.

---

## 3. Dtypes

MVP MUST support:

- `f32`

The optimized roadmap SHOULD support:

- `f16`
- `bf16`

Quantized formats MAY be added later.

Any reduced-precision mode MUST have a separate numerical tolerance and semantic decision regression suite.

---

## 4. Laya requirements

### 4.1 Checkpoint

Laya MUST accept existing checkpoint layouts compatible with the current Laya implementation, including:

- `model.safetensors`
- `rl_agent_config.json`
- `encoder/config.json`
- tokenizer assets

Runtime checkpoint conversion MUST NOT be mandatory.

### 4.2 Tokenizer

The Rust Laya engine MUST provide native tokenization.

It MUST NOT require Python at inference time.

Golden tests MUST compare token IDs, padding, truncation, special tokens and marker positions against the reference implementation.

### 4.3 Model

The implementation MUST cover:

- token embedding
- embedding LayerNorm
- ModernBERT encoder layers
- full attention
- sliding/local attention
- RoPE
- decision Transformer head
- type embeddings
- marker gather
- scorer
- action head
- calibration/result generation

### 4.4 Decisions

Laya MUST support:

- choice
- score
- noul

Question order and option order MUST be deterministic.

### 4.5 Attention mask

The optimized path SHOULD avoid allocating a dense `L × L` Boolean mask when the same semantics can be represented by:

- valid-token metadata
- window radius
- causal/local geometry

---

## 5. Jeff requirements

### 5.1 Initial supported subset

Jeff MUST initially target the same practical subset as the current native Julia path:

- text only
- Qwen3.5
- inference only
- prepared token IDs and masks
- bias-free projections where expected
- partial RoPE
- full attention
- Gated DeltaNet
- SiLU MLP
- Jeff readout

Generation and KV cache are NOT initial requirements.

### 5.2 Gated DeltaNet

The implementation MUST provide a correctness/reference path.

The optimized implementation MAY select between:

- recurrent formulation
- chunked formulation

Selection MUST be deterministic for the same model/configuration/runtime.

### 5.3 Readout

Qwen encoder logic and Jeff-specific decision readout MUST remain separately testable.

---

## 6. Jev requirements

`jev-client` MUST NOT depend on tenferro.

It SHOULD preserve the existing client security semantics:

- TLS verification
- fixed endpoint policy
- redirects disabled
- application-controlled retry
- response byte limit
- timeout budget
- max inflight
- content type validation
- credential redaction
- typed errors

---

## 7. Hot-path requirements

Steady-state inference SHOULD avoid:

- heap allocation
- dynamic dispatch in inner loops
- HashMap lookup
- string-based tensor lookup
- runtime graph construction
- runtime kernel compilation
- weight repacking
- repeated constant generation
- implicit device transfer
- per-op GPU synchronization

A small number of result/object allocations outside the numerical hot path MAY remain if profiling shows they are insignificant.

---

## 8. Workspace

Each inference context SHOULD own reusable scratch memory.

A workspace planner SHOULD reuse buffers with non-overlapping lifetimes.

GPU workspaces MUST respect asynchronous execution lifetime: a buffer MUST NOT be reused while queued device work may still access it.

---

## 9. Weight preparation

Weight transformation at load time MAY include:

- transpose
- packing
- dtype conversion
- QKV concatenation
- backend alignment
- device upload

Prepared weights MUST preserve model semantics within documented tolerance.

---

## 10. Attention

A reference attention implementation MUST exist.

Optimized CUDA attention SHOULD avoid materializing full score/probability matrices when practical.

Fused attention MAY combine:

- QKV split
- RoPE
- scaling
- masks
- online softmax
- V accumulation

Sliding attention SHOULD skip tiles outside the attention window.

---

## 11. Numerical behavior

The project MUST distinguish:

1. semantic equality
2. numerical agreement
3. bit identity

Bit identity MUST NOT be a universal requirement.

A backend MUST document `atol` and `rtol` used for validation.

Selected categorical decisions SHOULD match the reference across the regression suite.

---

## 12. Error behavior

Errors MUST be typed.

The implementation MUST NOT silently continue after:

- incompatible checkpoint shapes
- unsupported dtype
- unsupported backend operation
- invalid model configuration
- failed GPU upload
- device mismatch

---

## 13. Device transfer

CPU/GPU transfer MUST be explicit.

A CUDA tensor passed to unsupported CPU execution MUST produce an error unless the caller explicitly downloads it.

The same rule applies in reverse.

---

## 14. Concurrency

The implementation MUST document whether an inference context is `Send` and/or `Sync`.

Shared mutable scratch state MUST NOT be accessed concurrently without explicit synchronization.

Recommended model:

- immutable model may be shared
- each concurrent worker gets a separate inference context/workspace
- each CUDA context MAY own a stream

---

## 15. Performance acceptance

Every optimization PR SHOULD include:

- before/after benchmark
- numerical parity result
- workload description
- hardware/software metadata

An optimization that changes semantic output beyond accepted tolerance MUST NOT be accepted solely because it is faster.

---

## 16. Production v1.0 requirements

v1.0 MUST include:

- optimized Laya CPU
- optimized Laya CUDA
- optimized Jeff CPU
- optimized Jeff CUDA
- real checkpoint validation
- reproducible benchmark suite
- documented tolerances
- no silent device fallback
- no hidden host/device transfer
- stable workspace ownership
- license/third-party notices
