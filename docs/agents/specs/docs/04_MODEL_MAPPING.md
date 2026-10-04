# Model Mapping: Julia to Rust / tenferro

## 1. Laya.jl

Observed high-level structure:

```text
Tokenizer / prompt
    ↓
ModernBERT encoder
    ↓
typed embedding
    ↓
decision Transformer head
    ↓
marker gather
    ↓
scorer
    ↓
choice/score/noul + action
```

### Mapping table

| Julia Laya concept | Rust / tenferro plan |
|---|---|
| `Linear` | prepared GEMM / `dot_general` |
| `gather_columns` | gather/index-select |
| `LayerNorm` | `tenferro-infer` LayerNorm, later fused |
| `softmax` | stable reduction + exp + sum, later fused |
| `rope` | dedicated inference primitive |
| `qkv_attention` | reference composition then optimized attention ExtensionOp |
| `AttentionMask` | compact validity/window metadata where possible |
| residual | preallocated add / fused residual-normalization |
| GeGLU | fused inference kernel |
| marker gather | gather |
| action statistics | small device or host kernel depending profile |

### Important Laya optimization

The existing Julia Metal implementation demonstrates that performance benefits from:

- fused attention
- simdgroup normalization
- GeGLU fusion
- pooled device memory
- minimizing command buffers
- minimal downloads

The Rust/CUDA design should preserve the same principles, even if the exact implementation differs.

---

# 2. JeffClient.jl

Observed native model structure:

```text
Qwen3.5 embeddings
    ↓
hybrid layer stack
    ├── full attention
    └── linear attention / Gated DeltaNet
    ↓
final RMSNorm
    ↓
Jeff readout
```

Supported current-style inference scope:

- prepared token IDs and attention masks
- no generation
- no persistent KV cache
- Float32 reference path
- partial RoPE
- full attention
- Gated DeltaNet
- SiLU MLP
- readout

### Mapping table

| Julia Jeff concept | Rust / tenferro plan |
|---|---|
| native linear | prepared GEMM |
| embedding gather | gather |
| RMS normalization | `tenferro-infer` RMSNorm |
| partial RoPE | `tenferro-infer` Qwen RoPE variant |
| full attention | optimized attention path |
| causal masked softmax | fused/optimized inference primitive |
| depthwise conv1d | `tenferro-gated-delta` |
| recurrent Gated Delta | `tenferro-gated-delta` optimized kernel |
| chunk triangular solve | reference/fallback algorithm |
| SiLU MLP | fused gate |
| Jeff readout | separate, testable model head |

### Gated Delta strategy

Maintain two levels:

```text
reference:
tensor composition / explicit loops / solve

optimized:
CPU recurrent kernel
CUDA specialized kernel
```

The optimized implementation must remain numerically checked against the reference path.

---

# 3. JevClient.jl

Jev is fundamentally different.

Structure:

```text
QuestionSet
   ↓
serialization
   ↓
HTTPS request
   ↓
TypeSafe API
   ↓
response validation
   ↓
typed answers
```

tenferro has no meaningful role here.

Rust mapping:

| Julia Jev | Rust |
|---|---|
| `QuestionSet` | `decision-core` |
| `Client` | `jev-client::Client` |
| credential provider | secret-aware credential trait |
| HTTP transport | Rust HTTP/TLS client |
| retry policy | explicit retry layer |
| response size limits | streaming bounded body read |
| typed errors | Rust enums |
| mock transport | injectable test transport |

---

# 4. Shared typed-decision API

A common semantic interface is useful, but should not distort local model implementations.

Possible trait:

```rust
pub trait DecisionEngine {
    type Error;

    fn system_one(
        &mut self,
        state: &State,
        questions: &QuestionSet,
    ) -> Result<SystemOneResponse, Self::Error>;
}
```

This abstraction should be introduced only after Laya/Jeff model execution is stable.

---

# 5. Recommended implementation order

1. Shared `decision-core`
2. Inference primitives
3. Laya CPU
4. Laya CPU optimization
5. Laya CUDA
6. Jeff CPU reference
7. Gated Delta CPU optimization
8. Jeff CUDA
9. Jev Rust client
10. Apple GPU
