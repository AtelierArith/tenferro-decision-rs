# CPU and GPU Inference Design

## 1. Performance priority

The principal workload is low-latency, one-shot typed-decision inference.

Priority:

1. batch-1 latency
2. small-batch latency/throughput
3. memory efficiency
4. large-batch throughput

This differs from autoregressive LLM serving.

---

# 2. CPU inference

## 2.1 Matrix multiplication

Large dense projections should use the fastest validated tenferro CPU provider available for the platform.

Candidates include:

- faer-backed paths
- BLAS-backed paths
- platform-optimized BLAS where appropriate

Small projections must be benchmarked before automatically routing through a heavyweight BLAS call.

## 2.2 Elementwise fusion

Good CPU fusion targets:

- residual + LayerNorm
- residual + RMSNorm
- GeGLU
- SiLU gate
- softmax normalization loop
- DeltaNet recurrence

A fused native loop can reduce:

- allocations
- memory passes
- temporary arrays
- function dispatch overhead

## 2.3 SIMD

SIMD optimization SHOULD focus on memory-bound or reduction-heavy kernels:

- LayerNorm
- RMSNorm
- activation/gate
- softmax
- recurrent Delta state updates

GEMM SHOULD normally remain delegated to mature matrix libraries.

## 2.4 Threading

Nested parallelism must be controlled.

Bad configuration:

```text
outer Rayon threads
×
multithreaded BLAS
```

without an explicit budget.

The implementation should benchmark:

- single-thread end-to-end
- managed outer parallelism
- provider-owned BLAS parallelism

Batch-1 workloads may be faster with fewer threads because thread-launch/synchronization overhead matters.

## 2.5 CPU workspace

Steady-state model forward should reuse allocated buffers.

Example Laya workspace classes:

- hidden A
- hidden B / residual
- QKV
- attention output
- MLP intermediate
- decision-head scratch

Example Jeff workspace classes:

- hidden/residual
- projection scratch
- attention scratch
- DeltaNet state
- convolution scratch
- MLP intermediate

---

# 3. CUDA inference

## 3.1 Residency

After load:

- weights stay on GPU
- prepared constants stay on GPU
- RoPE tables may stay on GPU
- workspaces stay on GPU

Per request, only compact input and compact output should cross the PCIe/device boundary.

## 3.2 Launch minimization

CUDA performance work should minimize:

- kernel launch count
- global-memory intermediate writes
- stream synchronization
- repeated tensor canonicalization

## 3.3 Fused attention

Preferred execution:

```text
Q/K/V tile load
↓
RoPE/scaling
↓
masked dot products
↓
online softmax
↓
V accumulation
↓
output
```

Avoid:

```text
QK score allocation
↓
mask allocation
↓
softmax allocation
↓
P×V allocation
```

where a fused implementation is practical.

## 3.4 Online softmax

For attention, maintain running:

- maximum
- normalization sum
- weighted value accumulator

This avoids storing the entire probability matrix.

## 3.5 Laya local attention

The local/sliding attention kernel should exploit the window.

It should not visit all key tiles and then mask most of them.

Padding-only tiles should be skipped.

## 3.6 Jeff Gated Delta

The optimized CUDA Gated Delta kernel should keep per-head/sample recurrent state as close to the compute units as practical.

Preferred storage hierarchy:

1. registers where feasible
2. shared memory where feasible
3. global memory only when necessary

A token-by-token state round trip to global memory should be avoided if the state shape allows on-chip retention.

## 3.7 CUDA streams

For concurrent inference:

```text
InferenceContext A → CUDA stream A
InferenceContext B → CUDA stream B
```

The runtime should not require every independent request to serialize on a single process-global stream.

## 3.8 Synchronization

Latency measurement must include completion.

Do not report launch-only timing.

End-to-end GPU measurement should be:

```text
input ready
→ upload if included
→ kernels
→ result download/synchronize
→ output ready
```

---

# 4. Mixed precision

After FP32 correctness:

## FP16 / BF16

Candidates for reduced precision:

- projections
- attention inputs
- MLP

Sensitive reductions may continue to accumulate in FP32:

- LayerNorm/RMSNorm statistics
- softmax accumulators
- selected DeltaNet state calculations if required by stability

Backend precision policy must be explicit.

---

# 5. Quantization

Quantization is not required for initial releases.

Potential future path:

1. FP16/BF16
2. weight-only INT8
3. lower-precision formats where hardware support and model quality justify them

Every quantized configuration requires:

- logit comparison
- decision agreement
- latency measurement
- memory measurement
- representative data evaluation

---

# 6. Apple GPU

Apple GPU support should eventually use tenferro's Apple/WebGPU path or a compatible tenferro extension boundary.

Initial goals should be:

1. operation completeness
2. numerical parity
3. device-resident forward
4. then fusion/performance

Do not reproduce a second independent Metal abstraction inside each model crate unless it is a temporary experimental prototype.
