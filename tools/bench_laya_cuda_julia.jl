# Julia Laya.jl CUDA forward on the same fixed batches as tenferro's bench_laya_cuda.
# julia --project=<env with Laya, CUDA, JSON> tools/bench_laya_cuda_julia.jl CHECKPOINT [WARMUP ITERS]   (LAYA_BENCH_SHAPES=8x1,...)
using Laya, CUDA, JSON, Statistics
CUDA.device!(0); CUDA.allowscalar(false)
const DIR = ARGS[1]
const WARMUP = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
const ITERS = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 50
const BASE = Int32[2, 100, 1000, 2000, 3000, 4000, 5, 3]
shapes_spec = get(ENV, "LAYA_BENCH_SHAPES", "8x1,64x1,8x8")
SHAPES = [Tuple(parse.(Int, split(s, 'x'))) for s in split(shapes_spec, ',')]
t = time_ns(); model = Laya.load_backend_model(CUDABackend(), DIR, Float32); load_ms = (time_ns() - t) / 1e6
out = Any[]
for (len, batch) in SHAPES
    ids = reshape(Int32[BASE[mod1(l, length(BASE))] for _ in 1:batch for l in 1:len], len, batch)
    b = Dict("input_ids" => ids, "attention_mask" => trues(len, batch),
             "marker_pos" => repeat(reshape(Int32[1, 3, 5], 3, 1), 1, batch),
             "marker_mask" => trues(3, batch), "qtype" => zeros(Int32, batch))
    t = time_ns(); model(b); first_ms = (time_ns() - t) / 1e6
    for _ in 1:WARMUP; model(b); end
    s = [(t = time_ns(); model(b); (time_ns() - t) / 1e6) for _ in 1:ITERS]
    println(stderr, "L$(len)B$(batch): julia cuda median $(round(median(s); digits=2)) ms")
    push!(out, Dict("length" => len, "batch" => batch, "ms_median" => median(s), "ms_min" => minimum(s), "first_ms" => first_ms))
end
println(JSON.json(Dict("runtime" => "julia-laya-cuda", "load_ms" => load_ms, "warmup" => WARMUP, "iters" => ITERS, "shapes" => out)))
