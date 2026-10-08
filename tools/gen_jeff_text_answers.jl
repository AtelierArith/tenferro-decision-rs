# julia -t 8 --project=extern/JeffClient.jl tools/gen_jeff_text_answers.jl CHECKPOINT
# Run Julia decide on token ids captured from original Python Jeff.
using JeffClient, JSON
path = normpath(joinpath(@__DIR__, "..", "fixtures", "jeff-real", "tokenizer.json"))
reference = JSON.parsefile(path)
backend = NativeBackend(ARGS[1])
for case in reference["cases"]
    q = case["question"]
    question = if q["type"] == "choice"
        ChoiceQuestion([k => string(q["criteria"][k]) for k in case["criteria_order"]])
    elseif q["type"] == "score"
        ScoreQuestion(String.(q["criteria"]))
    else
        NoulQuestion()
    end
    ids = reshape(Int32.(case["input_ids"]), 1, :)
    case["julia_answer"] = decide(backend, Dict("input_ids" => ids, "attention_mask" => ones(Int, size(ids))), question)
end
reference["julia_version"] = string(VERSION)
write(path, JSON.json(reference))
println(path)
