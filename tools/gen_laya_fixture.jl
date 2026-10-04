# Generate the committed Laya fixture (`fixtures/laya-tiny/`) and its PyTorch-free
# Julia reference, for Rust cross-checks.
#
#   julia --project=extern/Laya.jl tools/gen_laya_fixture.jl
#
# The checkpoint weights are seeded, so the fixture and `reference.json` are
# reproducible. `reference.json` records a fixed prepared batch and its
# `DecisionModel` outputs (raw marker logits and action logits) plus a few
# tokenizer goldens, all produced by `extern/Laya.jl`.

using Laya, JSON, Random

const OUT = normpath(joinpath(@__DIR__, "..", "fixtures", "laya-tiny"))

Random.seed!(731)
Laya.write_tiny_checkpoint(
    OUT;
    hidden_size = 32,
    num_attention_heads = 2,
    intermediate_size = 48,
    num_hidden_layers = 2,
    head_layers = 1,
)

model, cfg, agent_cfg = Laya.load_model(OUT)
tok = Laya.Tokenizer(joinpath(OUT, "tokenizer"))

# A fixed, hand-built prepared batch (one row). Rust reconstructs exactly these
# arrays and compares the forward outputs.
ids = reshape(Int32[2, 5, 7, 11, 13, 4, 6, 3], 8, 1)
mask = trues(8, 1)
marker_pos = reshape(Int32[2, 4, 6], 3, 1)
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
write(joinpath(OUT, "reference.json"), JSON.json(reference))
println("act_count=", size(action, 1), " logits=", size(logits), " action=", size(action))
println("wrote ", OUT)
