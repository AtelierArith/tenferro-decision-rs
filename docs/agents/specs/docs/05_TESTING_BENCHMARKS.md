# Testing, Validation, Benchmarking and Profiling

## 1. Correctness levels

### Structural

Verify:

- shape
- dtype
- option ordering
- mask semantics
- selected answer type

### Numerical

Use:

```text
abs(actual - reference) <= atol + rtol * abs(reference)
```

Tolerance is backend/dtype specific.

### Semantic

Verify:

- same choice
- same noul truth decision
- score remains consistent
- calibrated ordering remains stable

Bit identity is not required universally.

---

# 2. Reference implementations

## Laya

Use at least one of:

- Laya.jl CPU
- Laya.jl Metal
- upstream/laya-mlx reference

## Jeff

Use at least one of:

- JeffClient.jl CPU
- JeffClient.jl Metal
- independent PyTorch Transformers reference

A highly optimized Rust path should be checked against a reference that does not share the same optimized kernel.

---

# 3. Intermediate activation tests

Final logits alone are insufficient for debugging.

## Laya checkpoints

Compare:

1. embeddings
2. embedding normalization
3. QKV projection
4. attention output
5. residual
6. MLP input/output
7. encoder final
8. decision head
9. marker values
10. logits
11. action output

## Jeff checkpoints

Compare:

1. embedding
2. RMSNorm
3. projection outputs
4. RoPE outputs
5. full attention
6. DeltaNet pre-convolution
7. convolution
8. delta recurrence/chunk result
9. MLP
10. final RMSNorm
11. Jeff readout logits

---

# 4. Shape regression suite

At minimum test:

```text
L = 1
L = 63
L = 64
L = 65
L = 127
L = 128
L = 129
L = 255
L = 256
L = 257
L = 512
```

Also cover:

- batch 1
- batch > 1
- left padding
- interior mask holes
- no padding
- one option
- many options
- local-attention window boundaries

---

# 5. Benchmark rules

Benchmark runs must state:

- git revision
- CPU/GPU model
- OS
- Rust version
- tenferro revision/version
- BLAS/provider
- thread count
- dtype
- batch size
- sequence length
- warmup count
- measured iteration count
- whether tokenization is included
- whether input upload is included
- whether output download/synchronization is included

---

# 6. Laya benchmark matrix

Reuse the practical workloads already used by Laya where possible:

| Workload | Meaning |
|---|---|
| short:1 | short state, one question |
| short:10 | short state, 10 questions |
| short:50 | short state, 50 questions |
| long:1 | near 512-token state, one question |
| long:10 | near 512-token state, 10 questions |

Report:

- p50
- p95
- min/max
- allocations
- peak memory
- output agreement

---

# 7. Jeff benchmark matrix

At minimum:

| Batch | Sequence |
|---:|---:|
| 1 | 128 |
| 1 | 256 |
| 1 | 512 |
| 4 | 256 |

Measure separately:

- padded full-length
- trimmed/active-length optimization if enabled

Do not compare two implementations using different effective workloads without prominently stating the difference.

---

# 8. Latency breakdown

Maintain at least four benchmark scopes:

```text
tokenization
model forward
decision/result formatting
end-to-end
```

For GPU also record:

```text
upload
compute
download/synchronize
```

---

# 9. CPU profiling

Primary tools:

- Linux `perf`
- flame graphs
- hardware counters

Investigate:

- cycles
- instructions
- IPC
- cache misses
- branch misses
- memory bandwidth
- vectorization
- BLAS behavior
- thread synchronization

---

# 10. CUDA profiling

Primary tools:

- Nsight Systems
- Nsight Compute

Investigate:

- launch gaps
- kernel count
- occupancy
- register pressure
- shared memory
- DRAM traffic
- achieved bandwidth
- Tensor Core use where applicable
- stream synchronization
- host/device transfer

---

# 11. Performance regression policy

Every performance-sensitive PR should contain:

```text
before
after
correctness status
hardware
workload
```

Recommended CI strategy:

- unit/correctness CI always
- stable CPU performance runner periodically
- CUDA performance runner periodically
- Apple GPU performance runner after Metal support matures

Noisy benchmark environments should not use overly strict single-run thresholds.

---

# 12. Allocation targets

For mature CPU steady-state inference:

- no repeated weight allocation
- no repeated model graph allocation
- no repeated shape-plan allocation
- minimal or zero large tensor heap allocation

For mature GPU steady-state inference:

- no device allocation in ordinary warm forward where practical
- persistent/recycled workspace buffers
- no intermediate host round trip

These are optimization targets, not prerequisites for the first correctness prototype.
