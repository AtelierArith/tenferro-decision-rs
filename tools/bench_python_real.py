#!/usr/bin/env python3
"""Benchmark original upstream PyTorch forwards on the Rust/Julia prepared inputs.

Requires a local upstream checkout and checkpoint; never downloads a model.
Example: python tools/bench_python_real.py laya CHECKPOINT --source LAYA_REPO
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("laya", "jeff"))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()
    if min(args.threads, args.warmup, args.iters) < 1:
        parser.error("threads, warmup, and iters must be positive")
    source = args.source.resolve()
    checkpoint = args.checkpoint.resolve()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(source / "src" if args.model == "jeff" else source))
    # Keep stdout machine-readable even if upstream loaders print progress.
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        import transformers
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        start = time.perf_counter()
        if args.model == "laya":
            import laya
            model = laya.load(str(checkpoint), device="cpu", backend="eager").model.eval()
            shapes = ((8, 1), (64, 1), (8, 8))
        else:
            from jeff.models import load_decision_model
            from jeff.model import PreparedBatch
            model = load_decision_model(checkpoint=checkpoint, device="cpu",
                                        cpu_threads=args.threads).eval()
            shapes = ((8, 1), (16, 1), (64, 1))
        load_ms = (time.perf_counter() - start) * 1000
        dtypes = sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()})
        if dtypes != ["torch.float32"]:
            raise RuntimeError(f"comparison requires float32 weights; found {dtypes}")
        base = [2, 100, 1000, 2000, 3000, 4000, 5, 3]
        reports = []
        reference_path = Path(__file__).resolve().parent.parent / "fixtures" / f"{args.model}-real" / "reference.json"
        reference = json.loads(reference_path.read_text())
        with torch.inference_mode():
            for length, batch in shapes:
                ids = torch.tensor([[base[i % 8] for i in range(length)]] * batch)
                if args.model == "laya":
                    inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                                  marker_pos=torch.tensor([[1, 3, 5]] * batch),
                                  marker_mask=torch.ones((batch, 3), dtype=torch.bool),
                                  qtype=torch.zeros(batch, dtype=torch.long))
                    def forward():
                        return model(**inputs)
                else:
                    count = model.readout.out_features
                    prepared = PreparedBatch(dict(input_ids=ids, attention_mask=torch.ones_like(ids)),
                                             (count,) * batch, length * batch)
                    def forward():
                        return (model(prepared),)
                for _ in range(args.warmup):
                    outputs = forward()
                if any(not torch.isfinite(output).all() for output in outputs):
                    raise RuntimeError("nonfinite upstream output")
                error = None
                if (length, batch) == (8, 1):
                    expected = torch.tensor(reference["logits"], dtype=torch.float32)
                    if args.model == "jeff":
                        expected = expected.unsqueeze(0)
                    torch.testing.assert_close(outputs[0], expected, atol=2e-4, rtol=2e-4)
                    error = float((outputs[0] - expected).abs().max())
                samples = []
                for _ in range(args.iters):
                    start = time.perf_counter()
                    outputs = forward()
                    samples.append((time.perf_counter() - start) * 1000)
                reports.append(dict(length=length, batch=batch,
                                    ms_median=statistics.median(samples), ms_min=min(samples),
                                    ms_mean=statistics.mean(samples), iterations=args.iters,
                                    samples_ms=samples, logits=outputs[0].tolist(),
                                    action_logits=outputs[1].tolist() if args.model == "laya" else None,
                                    max_reference_logit_error=error))
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                           text=True).strip()
    print(json.dumps(dict(runtime="original-python-pytorch-cpu", model=args.model,
                          source_revision=revision, checkpoint_revision=checkpoint.name,
                          source_dirty=bool(subprocess.check_output(
                              ["git", "-C", str(source), "status", "--porcelain"], text=True).strip()),
                          torch_version=torch.__version__, transformers_version=transformers.__version__,
                          python_version=platform.python_version(), platform=platform.platform(),
                          cpu_threads=torch.get_num_threads(), parameter_dtypes=dtypes,
                          warmup=args.warmup, load_ms=load_ms, shapes=reports), indent=2))


if __name__ == "__main__":
    main()
