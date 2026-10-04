# Source Snapshot Used for This Design

This document records the repository state and implementation facts considered while drafting the design.

Date of review: 2026-10-04

## Repositories

### Laya.jl

Repository:

```text
https://github.com/AtelierArith/Laya.jl
```

Observed default-branch source during review included commit:

```text
e9af31a4ccfe8692c2dbe8e7ddc456776f54df05
```

Relevant observed properties:

- pure Julia CPU model
- ModernBERT encoder
- decision Transformer head
- tokenizer and prompt builder
- safetensors support
- CPU backend
- Apple Accelerate support
- Metal backend
- separate MLX backend
- fused Metal attention / normalization / GeGLU optimizations
- choice / score / noul typed decisions

### JeffClient.jl

Repository:

```text
https://github.com/AtelierArith/JeffClient.jl
```

Observed source during review included commit:

```text
1338f4e7fc34a281e55d4ca5cf50565b9b91178b
```

Relevant observed properties:

- native Julia Qwen3.5 inference
- prepared token tensors
- safetensors loading
- CPU backend
- Metal backend
- CUDA backend
- partial RoPE
- full attention
- Gated DeltaNet
- RMS normalization
- SiLU MLP
- Jeff decision readout
- no persistent KV cache in the current decision-inference path
- text tokenization remains outside the native public inference path

### JevClient.jl

Repository:

```text
https://github.com/AtelierArith/JevClient.jl
```

Observed source during review included commit:

```text
4836ef382ddb5eb32e9dbd7e539880486beecaf9
```

Relevant observed properties:

- TypeSafe AI HTTP client
- choice / score / noul data model
- credential abstraction
- retries
- timeout/resource policies
- TLS verification
- endpoint restrictions
- typed response/error validation

It is not a tensor-compute package.

### tenferro-rs

Repository:

```text
https://github.com/tensor4all/tenferro-rs
```

Observed source during review included commit:

```text
67baa74b7114c08fcf81bd41cd3b7d791188ca24
```

Re-validated for implementation at:

```text
471c4278dbc5a955b8a1664789c37e5d07e743cb   (workspace version 0.7.1)
```

The `Cargo.toml` workspace pins `tenferro-runtime`, `tenferro-cpu`, and
`tenferro-tensor` to that git revision, because the crates.io `0.7.1` release
predates it and differs in the session-entry API
(`with_backend_session` returns a nested `Result` at `471c4278`). See
`11_TENFERRO_API_SURVEY.md`.

Relevant observed properties:

- CPU backend
- CUDA backend
- experimental WebGPU backend
- explicit device transfer model
- dense tensor operations
- reductions
- gather/scatter/slice
- `dot_general`
- einsum
- extension operations
- runtime extension modules
- CUDA coverage substantially broader than WebGPU coverage
- the practical WebGPU/Metal operation surface is close to `dot_general`
  (F32/C32) plus transpose/to-contiguous; elementwise, reductions, indexing, and
  linalg are not implemented — narrower than an earlier reading of this
  snapshot suggested (see `11_TENFERRO_API_SURVEY.md` §10)

---

## Design consequence

Based on the source snapshot above, the recommended order is:

```text
CPU
↓
CUDA
↓
Apple/Metal
```

for the Rust inference implementation.

The design should be revalidated if tenferro's WebGPU/Metal backend changes materially.
