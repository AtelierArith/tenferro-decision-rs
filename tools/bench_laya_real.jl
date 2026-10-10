# Benchmark the Julia Laya CPU runtime on the production checkpoint, using the
# same fixed prepared batches as the Rust `bench_laya` example.
#
#   julia --project=extern/Laya.jl tools/bench_laya_real.jl <CHECKPOINT_DIR> [WARMUP ITERS]
#
# Prints one JSON object on stdout. Model loading is measured and reported
# separately from the forward.

using Laya, JSON, Statistics, LinearAlgebra

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
    Dict(
        "ms_median" => median(samples),
        "ms_min" => minimum(samples),
        "samples_ms" => samples,
        "ms_mean" => mean(samples),
        "iterations" => iters,
    )
end

start = time_ns()
model, cfg, agent_cfg = Laya.load_model(DIR)
load_ms = (time_ns() - start) / 1e6

const BASE = Int32[2, 100, 1000, 2000, 3000, 4000, 5, 3]
const SHAPES = [(8, 1), (16, 1), (64, 1), (8, 8)]

shapes = Dict{String,Any}[]
for (len, batch) in SHAPES
    ids = reshape(Int32[BASE[mod1(l, length(BASE))] for _ in 1:batch for l in 1:len], len, batch)
    mask = trues(len, batch)
    marker_pos = repeat(reshape(Int32[1, 3, 5], 3, 1), 1, batch)
    marker_mask = trues(3, batch)
    qtype = zeros(Int32, batch)
    batch_dict = Dict(
        "input_ids" => ids,
        "attention_mask" => mask,
        "marker_pos" => marker_pos,
        "marker_mask" => marker_mask,
        "qtype" => qtype,
    )
    model(batch_dict) # compile/warm the first call
    stats = measure(() -> model(batch_dict), WARMUP, ITERS)
    stats["length"] = len
    stats["batch"] = batch
    push!(shapes, stats)
end

out = Dict(
    "runtime" => "julia-cpu",
    "checkpoint" => DIR,
    "load_ms" => load_ms,
    "warmup" => WARMUP,
    "shapes" => shapes,
    "julia_version" => string(VERSION),
    "cpu_model" => first(Sys.cpu_info()).model,
    "os" => string(Sys.KERNEL),
    "architecture" => string(Sys.ARCH),
    "blas_config" => string(BLAS.get_config()),
    "julia_threads" => Threads.nthreads(),
    "blas_threads" => BLAS.get_num_threads(),
)
println(JSON.json(out))
