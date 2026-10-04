# Benchmark the Julia Jeff native CPU runtime on the production checkpoint,
# using the same fixed prepared inputs as the Rust `bench_jeff` example.
#
#   julia --project=extern/JeffClient.jl tools/bench_jeff_real.jl <CHECKPOINT_DIR> [WARMUP ITERS]
#
# Prints one JSON object on stdout. Model loading is measured and reported
# separately from the forward. Inputs are single sequences of increasing length.

using JeffClient, JSON, Statistics

const DIR = ARGS[1]
const WARMUP = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
const ITERS = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 5

function measure(f::Function, warmup::Int, iters::Int)
    for _ in 1:warmup
        f()
    end
    samples = Float64[]
    for _ in 1:iters
        start = time_ns()
        f()
        push!(samples, (time_ns() - start) / 1e6)
    end
    sort!(samples)
    Dict(
        "ms_median" => median(samples),
        "ms_min" => first(samples),
        "ms_mean" => mean(samples),
        "iterations" => iters,
    )
end

start = time_ns()
backend = NativeBackend(DIR)
load_ms = (time_ns() - start) / 1e6

const BASE = Int32[2, 100, 1000, 2000, 3000, 4000, 5, 3]
const LENGTHS = [8, 16, 64]

shapes = Dict{String,Any}[]
for len in LENGTHS
    ids = reshape(Int32[BASE[mod1(i, length(BASE))] for i in 1:len], 1, len)
    mask = ones(Int, 1, len)
    inputs = Dict("input_ids" => ids, "attention_mask" => mask)
    logits(backend, inputs) # compile/warm the first call
    stats = measure(() -> logits(backend, inputs), WARMUP, ITERS)
    stats["length"] = len
    stats["batch"] = 1
    push!(shapes, stats)
end

out = Dict(
    "runtime" => "julia-cpu",
    "checkpoint" => DIR,
    "load_ms" => load_ms,
    "warmup" => WARMUP,
    "shapes" => shapes,
    "julia_version" => string(VERSION),
)
println(JSON.json(out))
