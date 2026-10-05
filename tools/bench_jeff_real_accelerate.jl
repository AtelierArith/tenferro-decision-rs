# Benchmark the Julia Jeff CPU runtime with Apple's Accelerate BLAS.
#
#   julia --project=<env with JeffClient + AppleAccelerate> tools/bench_jeff_real_accelerate.jl \
#       <CHECKPOINT_DIR> [WARMUP ITERS]
#
# Identical to tools/bench_jeff_real.jl except that AppleAccelerate is loaded
# first. Loading it forwards BLAS to Accelerate and triggers QwenDecisionCore's
# extension, which re-applies the CPU policy, so the Accelerate and default
# (OpenBLAS) paths can be measured side by side.

using AppleAccelerate
AppleAccelerate.load_accelerate()
include(joinpath(@__DIR__, "bench_jeff_real.jl"))
