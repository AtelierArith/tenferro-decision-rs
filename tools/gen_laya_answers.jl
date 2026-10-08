# julia -t 8 --project=extern/Laya.jl tools/gen_laya_answers.jl CHECKPOINT [OUT]
# Production answers and collated forward from the bundled upstream email example.
using Laya, JSON
root = normpath(joinpath(@__DIR__, ".."))
source = joinpath(root, "extern", "Laya.jl", "extern", "laya-mlx", "examples")
state = JSON.parsefile(joinpath(source, "state.json"))
questions = JSON.parsefile(joinpath(source, "questions.json"))
agent = Laya.load(ARGS[1])
items, internal = Laya.prepare(agent, state, questions)
batch = Laya.collate(items, agent.tok.pad_token_id)
logits, action = agent.model(batch)
prediction = Laya.predict(agent, state, questions)
reference = Dict(
    "state_entries" => [[k, v] for (k, v) in state],
    "question_ids" => collect(keys(questions)), "questions" => questions,
    "criteria_order" => [q.t == "choice" ? first.(q.crit) : nothing for q in internal],
    "items" => [Dict("ids" => it.ids, "markers" => it.markers, "qtype" => it.qtype) for it in items],
    "batch" => Dict(k => vec(v) for (k, v) in batch),
    "length" => size(batch["input_ids"], 1), "slots" => size(logits, 1),
    "logits" => vec(logits), "action" => vec(action), "prediction" => prediction,
    "checkpoint_revision" => basename(normpath(ARGS[1])), "julia_version" => string(VERSION),
)
output = length(ARGS) >= 2 ? ARGS[2] : joinpath(root, "fixtures", "laya-real", "answers.json")
write(output, JSON.json(reference))
println("batch=", size(batch["input_ids"]), " -> ", output)
