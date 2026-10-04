# Generate the production-Laya reference used by
# `crates/laya-infer/tests/real_checkpoint.rs`.
#
#   julia --project=extern/Laya.jl tools/gen_laya_real_reference.jl <CHECKPOINT_DIR> [OUT_JSON]
#
# `CHECKPOINT_DIR` is a resolved `convaiinnovations/laya` snapshot (see
# `hf-fetch laya`). The reference records a fixed prepared batch and its
# `DecisionModel` outputs, plus a few tokenizer goldens.

using Laya, JSON

const DIR = length(ARGS) >= 1 ? ARGS[1] :
    error("usage: gen_laya_real_reference.jl <CHECKPOINT_DIR> [OUT_JSON]")
const OUT = length(ARGS) >= 2 ? ARGS[2] :
    normpath(joinpath(@__DIR__, "..", "fixtures", "laya-real", "reference.json"))

model, cfg, agent_cfg = Laya.load_model(DIR)
tok = Laya.Tokenizer(joinpath(DIR, "tokenizer"))

ids = reshape(Int32[2, 100, 1000, 2000, 3000, 4000, 5, 3], 8, 1)
mask = trues(8, 1)
marker_pos = reshape(Int32[1, 3, 5], 3, 1)
marker_mask = trues(3, 1)
qtype = Int32[0]
batch = Dict(
    "input_ids" => ids,
    "attention_mask" => mask,
    "marker_pos" => marker_pos,
    "marker_mask" => marker_mask,
    "qtype" => qtype,
)
logits, action = model(batch)

samples = ["hello world", "it's a test", "café", "日本語"]
token_ids = [collect(Int32, tok(s)) for s in samples]

reference = Dict(
    "input_ids" => ids,
    "attention_mask" => mask,
    "marker_pos" => marker_pos,
    "marker_mask" => marker_mask,
    "qtype" => qtype,
    "logits" => logits,
    "action" => action,
    "tokenizer_samples" => samples,
    "tokenizer_ids" => token_ids,
    "hidden_size" => cfg.hidden_size,
    "vocab_size" => cfg.vocab_size,
    "act_count" => size(action, 1),
    "head_layers" => agent_cfg["head_layers"],
)
mkpath(dirname(OUT))
write(OUT, JSON.json(reference))
println("logits=", size(logits), " action=", size(action), " -> ", OUT)
