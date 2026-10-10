#!/usr/bin/env python3
"""Render the comparison table for tools/bench_compare.sh.

Reads the four raw benchmark JSON files from a results directory and prints the
Julia-vs-Rust (host and tenferro) comparison as Markdown. Not meant to be run
directly; bench_compare.sh calls it after the benchmark runs.
"""

import json
import os
import sys


def main() -> None:
    if len(sys.argv) != 5:
        sys.exit("usage: bench_table.py RESULTS_DIR THREADS WARMUP ITERS")
    out, threads, warmup, iters = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    def load(name):
        with open(os.path.join(out, name), encoding="utf-8") as handle:
            return json.load(handle)

    rust_jeff = load("rust_jeff.json")
    julia_jeff = load("julia_jeff.json")
    rust_laya = load("rust_laya.json")
    julia_laya = load("julia_laya.json")
    julia_jeff_acc_path = os.path.join(out, "julia_jeff_accelerate.json")
    julia_jeff_acc = load("julia_jeff_accelerate.json") if os.path.exists(julia_jeff_acc_path) else None
    julia_laya_acc_path = os.path.join(out, "julia_laya_accelerate.json")
    julia_laya_acc = load("julia_laya_accelerate.json") if os.path.exists(julia_laya_acc_path) else None

    def by_shape(report, length, batch):
        for shape in report["shapes"]:
            if shape["length"] == length and shape["batch"] == batch:
                return shape
        return None

    def ms(value):
        return f"{value:.1f}" if value is not None else "—"

    def ratio(a, b):
        return f"{a / b:.2f}×" if (a is not None and b) else "—"

    def optional_median(report, key):
        return (report.get(key) or {}).get("ms_median")

    def label(length, batch):
        return f"L{length}" if batch == 1 else f"L{length} B{batch}"

    julia_threads = julia_jeff.get("julia_threads", threads)
    blas_threads = julia_jeff.get("blas_threads", "?")
    laya_blas_threads = julia_laya.get("blas_threads", "?")

    print()
    print("## Benchmarks")
    print()
    print(
        f"Machine-local CPU comparison (`tools/bench_compare.sh`), "
        f"{threads} threads both sides: Julia `-t{threads}`, Rust "
        f"`RAYON_NUM_THREADS={threads}`. Warmup {warmup}, {iters} iterations, "
        f"median. Julia reported `julia_threads={julia_threads}`, "
        f"`blas_threads={blas_threads}` (Jeff) / `{laya_blas_threads}` (Laya)."
    )
    print()

    # Report the CPU paths selected by the production engines, separately from
    # the reference and diagnostic implementations below.
    prepared_laya_path = os.path.join(out, "rust_laya_prepared.json")
    if os.path.isfile(prepared_laya_path):
        prepared_laya = load("rust_laya_prepared.json")
        print("### Production CPU forwards (ms, median)")
        print()
        print("| model | shape | Julia | Rust production | Rust / Julia |")
        print("|---|---|---:|---:|---:|")
        for shape in rust_jeff["shapes"]:
            length, batch = shape["length"], shape["batch"]
            j = by_shape(julia_jeff, length, batch)
            rm = shape.get("prepared_cpu", {}).get("ms_median")
            jm = j["ms_median"] if j else None
            print(f"| Jeff | {label(length, batch)} | {ms(jm)} | {ms(rm)} | {ratio(rm, jm)} |")
        for shape in prepared_laya["rows"]:
            length, batch = shape["length"], shape["batch"]
            j = by_shape(julia_laya, length, batch)
            jm = j["ms_median"] if j else None
            rm = shape["tenferro_ms"]
            print(f"| Laya | {label(length, batch)} | {ms(jm)} | {ms(rm)} | {ratio(rm, jm)} |")
        print()

    for name, rust, julia in (("jeff", rust_jeff, julia_jeff), ("laya", rust_laya, julia_laya)):
        filename = f"python_{name}.json"
        if not os.path.isfile(os.path.join(out, filename)):
            continue
        upstream = load(filename)
        print(f"### {name.title()} — original Python CPU forward (ms, median)")
        print()
        print(f"PyTorch {upstream['torch_version']}, Transformers {upstream['transformers_version']}, "
              f"source `{upstream['source_revision']}`, {upstream['cpu_threads']} threads, float32.")
        print()
        print("| shape | Python | Julia | Rust host |")
        print("|---|---:|---:|---:|")
        for shape in upstream["shapes"]:
            length, batch = shape["length"], shape["batch"]
            r, j = by_shape(rust, length, batch), by_shape(julia, length, batch)
            rm = r.get("host_opt", {}).get("ms_median", r.get("ms_median")) if r else None
            jm = j.get("ms_median") if j else None
            print(f"| {label(length, batch)} | {ms(shape['ms_median'])} | {ms(jm)} | {ms(rm)} |")
        print()

    # ---- Jeff: forward latency ----
    print("### Jeff — forward (ms, median)")
    print()
    if julia_jeff_acc is not None:
        print("| length | Julia (OpenBLAS) | Julia (Accelerate) | Rust oracle | Rust host_opt | host_opt / Julia (best) |")
        print("|---|---:|---:|---:|---:|---:|")
        for length in (8, 16, 64):
            j = by_shape(julia_jeff, length, 1)
            a = by_shape(julia_jeff_acc, length, 1)
            r = by_shape(rust_jeff, length, 1)
            jm = j["ms_median"] if j else None
            am = a["ms_median"] if a else None
            om = r["ms_median"] if r else None
            hm = r["host_opt"]["ms_median"] if r and "host_opt" in r else None
            best = min(m for m in (jm, am) if m is not None) if (jm or am) else None
            print(
                f"| {label(length, 1)} | {ms(jm)} | {ms(am)} | {ms(om)} | {ms(hm)} | "
                f"{ratio(hm, best)} |"
            )
    else:
        print("| length | Julia | Rust oracle | Rust host_opt | host_opt / Julia |")
        print("|---|---:|---:|---:|---:|")
        for length in (8, 16, 64):
            j = by_shape(julia_jeff, length, 1)
            r = by_shape(rust_jeff, length, 1)
            jm = j["ms_median"] if j else None
            om = r["ms_median"] if r else None
            hm = r["host_opt"]["ms_median"] if r and "host_opt" in r else None
            print(
                f"| {label(length, 1)} | {ms(jm)} | {ms(om)} | {ms(hm)} | "
                f"{ratio(hm, jm)} |"
            )
    print()

    # ---- Jeff: tenferro vs host ----
    host8 = by_shape(rust_jeff, 8, 1)
    host8_ms = host8["host_opt"]["ms_median"] if host8 and "host_opt" in host8 else None
    j8 = by_shape(julia_jeff, 8, 1)
    j8_ms = j8["ms_median"] if j8 else None
    if julia_jeff_acc is not None:
        a8 = by_shape(julia_jeff_acc, 8, 1)
        if a8 is not None:
            j8_ms = min(v for v in (j8_ms, a8["ms_median"]) if v is not None)
    jeff_tenferro = [
        ("Rust host_opt (L8)", host8_ms),
        ("Rust tenferro `HostRecurrent` (cached)", optional_median(rust_jeff, "tenferro_cached_forward_8")),
        ("Rust tenferro `TensorNative` (cached)", optional_median(rust_jeff, "tenferro_native_forward_8")),
        ("Rust tenferro `HostRecurrent` (fresh cache)", optional_median(rust_jeff, "tenferro_forward_8")),
    ]
    print("### Jeff L8 — Rust host vs Rust tenferro")
    print()
    print("| path | ms | vs host_opt | vs Julia |")
    print("|---|---:|---:|---:|")
    for name, value in jeff_tenferro:
        print(f"| {name} | {ms(value)} | {ratio(value, host8_ms)} | {ratio(value, j8_ms)} |")
    print()

    # ---- Laya: forward latency ----
    print("### Laya — forward (ms, median)")
    print()
    if julia_laya_acc is not None:
        print("| shape | Julia (OpenBLAS) | Julia (Accelerate) | Rust host | Rust host / Julia (best) |")
        print("|---|---:|---:|---:|---:|")
        for length, batch in ((8, 1), (64, 1), (8, 8)):
            j = by_shape(julia_laya, length, batch)
            a = by_shape(julia_laya_acc, length, batch)
            r = by_shape(rust_laya, length, batch)
            jm = j["ms_median"] if j else None
            am = a["ms_median"] if a else None
            rm = r["ms_median"] if r else None
            best = min(m for m in (jm, am) if m is not None) if (jm or am) else None
            print(
                f"| {label(length, batch)} | {ms(jm)} | {ms(am)} | {ms(rm)} | "
                f"{ratio(rm, best)} |"
            )
    else:
        print("| shape | Julia | Rust host | host / Julia |")
        print("|---|---:|---:|---:|")
        for length, batch in ((8, 1), (64, 1), (8, 8)):
            j = by_shape(julia_laya, length, batch)
            r = by_shape(rust_laya, length, batch)
            jm = j["ms_median"] if j else None
            rm = r["ms_median"] if r else None
            print(f"| {label(length, batch)} | {ms(jm)} | {ms(rm)} | {ratio(rm, jm)} |")
    print()

    # ---- Laya: tenferro vs host ----
    laya_host = by_shape(rust_laya, 8, 1)
    laya_host_ms = laya_host["ms_median"] if laya_host else None
    laya_j = by_shape(julia_laya, 8, 1)
    laya_j_ms = laya_j["ms_median"] if laya_j else None
    if julia_laya_acc is not None:
        laya_ja = by_shape(julia_laya_acc, 8, 1)
        if laya_ja is not None:
            laya_j_ms = min(v for v in (laya_j_ms, laya_ja["ms_median"]) if v is not None)
    laya_tenferro = [
        ("Rust host (L8 B1)", laya_host_ms),
        ("Rust tenferro (cached)", rust_laya.get("tenferro_cached_forward_8x1", {}).get("ms_median")),
        ("Rust tenferro (fresh cache)", rust_laya.get("tenferro_forward_8x1", {}).get("ms_median")),
    ]
    print("### Laya L8 B1 — Rust host vs Rust tenferro")
    print()
    print("| path | ms | vs host | vs Julia |")
    print("|---|---:|---:|---:|")
    for name, value in laya_tenferro:
        print(f"| {name} | {ms(value)} | {ratio(value, laya_host_ms)} | {ratio(value, laya_j_ms)} |")
    print()

    # ---- model load ----
    print("### Model load (ms)")
    print()
    print("| model | Julia | Rust | Rust / Julia |")
    print("|---|---:|---:|---:|")
    for name, jr, rr in (
        ("Jeff", julia_jeff, rust_jeff),
        ("Laya", julia_laya, rust_laya),
    ):
        jl = jr.get("load_ms")
        rl = rr.get("load_ms")
        print(f"| {name} | {ms(jl)} | {ms(rl)} | {ratio(rl, jl)} |")
    print()


if __name__ == "__main__":
    main()
