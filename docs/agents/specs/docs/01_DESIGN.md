# Design Document

## 1. Objective

Build a Rust inference stack for Laya and Jeff that is optimized for low latency and predictable execution, while preserving the typed-decision semantics of the existing Julia implementations.

The design deliberately distinguishes:

1. **model semantics**
2. **inference kernels**
3. **backend/runtime ownership**
4. **HTTP client functionality**

This separation prevents model code from becoming entangled with backend-specific implementation details.

---

## 2. System architecture

```text
                         Application
                              │
                       decision-core
                              │
             ┌────────────────┼────────────────┐
             │                │                │
         laya-infer       jeff-infer        jev-client
             │                │                │
             └─────────┬──────┘                │
                       │                       HTTPS
                 tenferro-infer
                       │
              tenferro-gated-delta
                       │
                   tenferro-rs
                       │
                ┌──────┴──────┐
                │             │
               CPU          CUDA
```

Apple/Metal is deliberately outside the first critical path.

---

## 3. Design principles

### 3.1 Inference only

The runtime must not carry training machinery that is unused by production inference.

Not required:

- autodiff
- backward
- JVP/VJP
- optimizer state
- parameter updates
- generic trainable layer API
- dynamic training graph

Inference binaries should avoid linking AD functionality unless explicitly required by a future experimental feature.

### 3.2 Performance over generality

Known model structure should be exploited.

Permitted optimizations include:

- fixed model topology
- shape bucketing
- packed weights
- precomputed RoPE tables
- static workspace plans
- fused kernels
- backend-specific prepared plans
- model-specific ExtensionOps

The hot path should not pay abstraction costs merely to support models that are outside scope.

### 3.3 Correct reference path first

Each optimized kernel must have a simpler reference implementation.

Recommended progression:

```text
correct tensor composition
        ↓
reference parity
        ↓
profiling
        ↓
fusion / specialization
        ↓
optimized backend kernel
```

### 3.4 No silent fallback

A GPU workload must not silently transfer to CPU because an operation is unsupported.

Unsupported device operations return explicit typed errors.

### 3.5 Explicit ownership

Long-lived mutable state must have a visible owner:

- model owns immutable weights and static plans
- inference context owns scratch/workspace
- backend/runtime owns device queues, streams, memory pools and kernel caches
- Jev client owns connection/retry/credential state

No unbounded process-global mutable cache.

---

## 4. Crate responsibilities

### decision-core

Owns backend-independent typed decision data:

- `QuestionSet`
- `Noul`
- `Choice`
- `Score`
- answer types
- usage metadata
- optional common `DecisionEngine` trait

It does **not** depend on tenferro.

### tenferro-infer

Owns inference-specific reusable operations needed by Laya and Jeff:

- Linear helpers / prepared GEMM interfaces
- embedding gather
- LayerNorm
- RMSNorm
- GELU / GeGLU
- SiLU / gated SiLU
- softmax / masked softmax
- RoPE variants
- residual + normalization fusions
- attention
- layout helpers specific to inference

This is not a generic NN framework.

### tenferro-gated-delta

Owns Jeff/Qwen3.5 Gated DeltaNet execution:

- causal depthwise convolution
- recurrent delta formulation
- chunked/reference formulation
- CPU optimized implementation
- CUDA specialized implementation

### laya-infer

Owns:

- Laya checkpoint loading
- tokenizer
- prompt building
- ModernBERT model
- decision head
- calibration
- Laya-specific execution plans

### jeff-infer

Owns:

- Jeff checkpoint loading
- supported Qwen3.5 subset
- full-attention layers
- Gated DeltaNet layers
- Jeff readout
- prepared-token inference
- later optional tokenizer support

### jev-client

Owns:

- TypeSafe API transport
- credentials
- retries/timeouts
- resource limits
- response parsing
- endpoint policy

It must not depend on tenferro.

---

## 5. Model/context split

For fast inference, immutable model state and mutable per-execution state should be separable.

Example:

```rust
pub struct LayaModel {
    weights: Arc<LayaWeights>,
    plan: Arc<LayaPlan>,
}

pub struct LayaContext {
    model: Arc<LayaModel>,
    workspace: LayaWorkspace,
}
```

Benefits:

- weights can be shared
- workspaces do not require locks
- concurrent requests can use separate contexts
- CUDA streams can be context-owned
- allocation lifetime is explicit

The same pattern should be usable for Jeff.

---

## 6. Load-time vs inference-time work

### Load time

Perform as much reusable work as possible:

- parse config
- validate tensor names/shapes
- read safetensors
- convert dtype if requested
- pack weights
- upload weights
- choose algorithms
- create runtime/backend instances
- allocate workspaces
- compile/prepare kernels
- prepare mask metadata
- prepare head mappings
- prepare RoPE frequency/tables
- construct shape-specialized plans

### Inference time

The ideal inference path is close to:

```text
copy compact input
launch prepared kernels
download tiny result
format result
```

The following should not occur per request unless unavoidable:

- checkpoint parsing
- tensor-name string lookup
- weight transposition
- general graph optimization
- kernel compilation
- RoPE frequency generation
- allocator churn
- hidden host/device synchronization

---

## 7. Tensor layout policy

A model should have a documented canonical runtime layout.

Checkpoint layout is an input format, not necessarily the optimal compute layout.

At load time, the runtime may:

- transpose weights
- pack weights
- align storage
- merge Q/K/V matrices
- convert to device-optimal layout

Hot-path layout conversion should be minimized.

---

## 8. Prepared execution

The project should prefer a prepared inference pipeline.

For common sequence lengths, plans may be bucketed:

```text
64
128
256
512
```

A generic fallback remains required for correctness, but commonly measured production shapes should avoid repeated planning.

---

## 9. Fusion policy

Fusion is strongly encouraged when it reduces:

- global-memory traffic
- intermediate allocation
- GPU launch count
- synchronization
- repeated normalization passes

Candidate fusions:

### Laya

- residual + LayerNorm
- QKV split + RoPE + attention + online softmax + V accumulation
- GeGLU
- small decision-head operations where profiling proves launch overhead matters

### Jeff

- residual + RMSNorm
- Q/K normalization + partial RoPE + grouped-KV preparation
- causal softmax
- depthwise convolution + Gated Delta recurrence where beneficial
- SiLU gate

Fusion must follow reference-correctness validation.

---

## 10. Why not one giant model kernel?

The entire Laya or Qwen3.5 model must not become a single opaque ExtensionOp.

That would make:

- correctness isolation difficult
- backend evolution difficult
- testing difficult
- useful reusable primitives unavailable

The preferred granularity is:

```text
reusable high-value inference primitive
+
model-specific execution plan
```

---

## 11. Apple GPU policy

The long-term architecture should permit Apple GPU support through tenferro's WebGPU/Metal path.

However, the first production release should target:

1. CPU
2. CUDA
3. Apple/Metal later

The model crates must therefore avoid hardcoding CUDA assumptions while still allowing CUDA-specific optimized ExtensionOps behind the backend boundary.
