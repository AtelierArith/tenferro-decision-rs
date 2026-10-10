# Julia JeffClient/QwenDecisionCore CUDA on the same inputs as tenferro's bench_jeff_cuda.
# julia --project=<env with JeffClient, CUDA, JSON> tools/bench_jeff_cuda_julia.jl CHECKPOINT PARCEL_JSON [WARMUP ITERS]
using JeffClient, CUDA, JSON, Statistics
CUDA.device!(0); CUDA.allowscalar(false)
const DIR, PARCEL = ARGS[1], ARGS[2]
const WARMUP = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 10
const ITERS = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 50
const BASE = [2, 100, 1000, 2000, 3000, 4000, 5, 3]
matrix(rows) = reduce(vcat, [permutedims(Int64.(r)) for r in rows])
t = time_ns(); backend = NativeBackend(DIR; device = :cuda); load_ms = (time_ns() - t) / 1e6
ref = JSON.parsefile(PARCEL); case = (ref isa AbstractDict ? ref["cases"] : ref)[1]
parcel = Dict(k => matrix(v) for (k, v) in case["inputs"])
synthetic(len) = Dict("input_ids" => reshape([BASE[mod1(i, 8)] for i in 1:len], 1, len), "attention_mask" => ones(Int64, 1, len))
cases = [("parcel_L256_active101", parcel, "1"), ("synthetic_L8", synthetic(8), "1"),
         ("synthetic_L64", synthetic(64), "1"), ("parcel_full_L256", parcel, "0")]
out = Any[]
for (name, inputs, trim) in cases
    ENV["QDC_CUDA_TRIM_PADDING"] = trim   # read per call by QwenDecisionCore
    t = time_ns(); logits(backend, inputs); first_ms = (time_ns() - t) / 1e6
    for _ in 1:WARMUP; logits(backend, inputs); end
    s = [(t = time_ns(); logits(backend, inputs); (time_ns() - t) / 1e6) for _ in 1:ITERS]
    println(stderr, "$name: julia cuda median $(round(median(s); digits=2)) ms")
    push!(out, Dict("name" => name, "ms_median" => median(s), "ms_min" => minimum(s), "first_ms" => first_ms))
end
println(JSON.json(Dict("runtime" => "julia-jeff-cuda", "load_ms" => load_ms, "warmup" => WARMUP, "iters" => ITERS, "cases" => out)))
