# Generate the production-Jeff reference used by
# `crates/jeff-infer/tests/real_checkpoint.rs`.
#
#   julia --project=extern/JeffClient.jl tools/gen_jeff_real_reference.jl <CHECKPOINT_DIR> [OUT_JSON]
#
# `CHECKPOINT_DIR` is a resolved `mstrasser/Jeff-Qwen3.5-0.8B` snapshot (see
# `hf-fetch jeff`). The reference records the uncalibrated readout logits for a
# fixed single sequence.

using JeffClient, JSON

const DIR = length(ARGS) >= 1 ? ARGS[1] :
    error("usage: gen_jeff_real_reference.jl <CHECKPOINT_DIR> [OUT_JSON]")
const OUT = length(ARGS) >= 2 ? ARGS[2] :
    normpath(joinpath(@__DIR__, "..", "fixtures", "jeff-real", "reference.json"))

backend = NativeBackend(DIR)

const BASE = Int32[2, 100, 1000, 2000, 3000, 4000, 5, 3]
ids = reshape(BASE, 1, 8)
mask = ones(Int, 1, 8)
inputs = Dict("input_ids" => ids, "attention_mask" => mask)
scores = logits(backend, inputs) # (1, options)

reference = Dict(
    "input_ids" => collect(Int64, vec(ids)),
    "attention_mask" => vec(mask),
    "logits" => vec(scores),
    "options" => size(scores, 2),
)
# Each question consumes one row; include interior holes and left padding.
answer_ids = repeat(ids, 3, 1)
answer_mask = ones(Int, 3, 8)
answer_mask[2, 2] = 0
answer_mask[3, 1:2] .= 0
questions = [
    ChoiceQuestion(["a" => "first", "b" => "second", "c" => "third"]; instructions="prepared-token question"),
    NoulQuestion(instructions="prepared-token question"),
    ScoreQuestion(["low", "middle", "high"]; instructions="prepared-token question"),
]
answers = decide(backend, Dict("input_ids" => answer_ids, "attention_mask" => answer_mask), questions)
reference["answer_cases"] = [
    Dict("input_ids" => collect(answer_ids[i, :]),
         "attention_mask" => collect(answer_mask[i, :]), "answer" => answers[i])
    for i in eachindex(questions)
]
reference["temperature"] = backend.temperature
reference["checkpoint_revision"] = basename(normpath(DIR))
reference["julia_version"] = string(VERSION)
mkpath(dirname(OUT))
write(OUT, JSON.json(reference))
println("logits=", size(scores), " -> ", OUT)
