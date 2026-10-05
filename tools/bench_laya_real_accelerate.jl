# Benchmark Laya's CPU forward with Apple's Accelerate BLAS (Apple silicon only).
#
#   julia --project=<env with Laya + AppleAccelerate> tools/bench_laya_real_accelerate.jl \
#       <CHECKPOINT_DIR> [WARMUP ITERS]
#
# Identical to tools/bench_laya_real.jl except that BLAS is forwarded to Accelerate
# before the checkpoint is loaded. Laya's default `CPUBackend` then runs its matrix
# products through Accelerate process-wide (see Laya's `AccelerateBackend`).

using AppleAccelerate
AppleAccelerate.load_accelerate()
include(joinpath(@__DIR__, "bench_laya_real.jl"))
