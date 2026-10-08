#!/usr/bin/env python3
"""Capture original Jeff prompts/token ids without loading model weights.
python tools/gen_jeff_tokenizer_reference.py CHECKPOINT [OUTPUT]
"""
import json
import os
from pathlib import Path
import sys
os.environ["HF_HUB_OFFLINE"] = "1"
root = Path(__file__).resolve().parent.parent
source = root / "extern/JeffClient.jl/extern/jeff"
sys.path.insert(0, str(source / "src"))
from jeff.model import decision_messages
from transformers import AutoProcessor
checkpoint = Path(sys.argv[1])
processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
config = json.loads((checkpoint / "decision_config.json").read_text())
cases = []
for state, question in [
    ("The package arrived late. Café 日本語 e\u0301.",
     {"type":"choice", "instructions":"Select a response.", "criteria":{"reply":"Apologize", "wait":None, "refund":{"amount":1.0}}}),
    ({"customer":"日本語", "amount":1e-7, "active":True},
     {"type":"score", "instructions":"Rate urgency.", "criteria":["low", "middle", "high"]}),
    ("Please refund the duplicate charge.",
     {"type":"noul", "instructions":"Is a refund requested?", "criteria":{"false":"", "true":"Yes / true"}}),
]:
    messages = decision_messages({"state":state, "question":question}, config["codes"], config.get("prompt_layout", "state-first"))
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=[prompt], return_tensors="pt")
    cases.append(dict(state=state, state_order=list(state) if isinstance(state, dict) else None,
                      question=question, criteria_order=list(question["criteria"]) if question["type"]=="choice" else None,
                      prompt=prompt, input_ids=inputs["input_ids"][0].tolist()))
report = dict(checkpoint_revision=checkpoint.name, cases=cases,
              samples=[{"text":t,"ids":processor.tokenizer.encode(t, add_special_tokens=False)}
                       for t in ["it's a test", "café e\u0301 日本語", "1 23 456\n", "<|im_start|>user\nhello"]])
output = Path(sys.argv[2]) if len(sys.argv)>2 else root/"fixtures/jeff-real/tokenizer.json"
output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
print(output)
