#!/usr/bin/env bash
#
# Reproducible CPU benchmark comparison: Julia vs Rust, for the Jeff and Laya
# decision models. Rust is measured on both its host forward and its
# tenferro-native forward, so the three paths sit side by side.
#
# It drives the existing per-runtime benchmarks and renders one comparison
# table at the end:
#
#   Julia              tools/bench_jeff_real.jl / tools/bench_laya_real.jl
#   Rust host          target/release/examples/bench_jeff   (host_opt column)
#   Rust tenferro      target/release/examples/bench_jeff   (tenferro_* keys)
#
# Usage:
#
#   tools/bench_compare.sh --jeff <JEFF_CKPT_DIR> --laya <LAYA_CKPT_DIR> \
#       [--threads N] [--warmup W] [--iters I] [--no-build] [--json DIR] \
#       [--acc-env DIR]
#
#   JEFF_CKPT_DIR  mstrasser/Jeff-Qwen3.5-0.8B snapshot (decision_config.json +
#                  readout.safetensors + model.safetensors)
#   LAYA_CKPT_DIR  convaiinnovations/laya snapshot (encoder/ + tokenizer/ +
#                  rl_agent_config.json + model.safetensors)
#
# Threads default to 8: Julia is launched with `-t N` and Rust with
# `RAYON_NUM_THREADS=N`. Run the benchmarks one at a time on an idle machine.
#
# --json DIR keeps the raw JSON from every run in DIR instead of a temp dir.
#
# --acc-env DIR adds the Apple-silicon Julia runs for both models with BLAS
#   forwarded to Accelerate (Julia's fastest CPU path). DIR must be a Julia
#   project that has `JeffClient`, `Laya`, and `AppleAccelerate`; create one
#   with:
#
#     ENV=$(mktemp -d)
#     julia --project="$ENV" -e 'using Pkg;
#         Pkg.develop(path="extern/JeffClient.jl");
#         Pkg.develop(path="extern/Laya.jl");
#         Pkg.add(["AppleAccelerate", "JSON"])'
#     tools/bench_compare.sh ... --acc-env "$ENV"
#
#   On non-arm64 macOS the flag is ignored with a warning.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

THREADS=8
WARMUP=5
ITERS=30
BUILD=1
OUT=""
JEFF=""
LAYA=""
ACC_ENV=""

usage() {
    sed -n '2,41p' "$ROOT/tools/bench_compare.sh" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --jeff)    JEFF="$2"; shift 2 ;;
        --laya)    LAYA="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --warmup)  WARMUP="$2"; shift 2 ;;
        --iters)   ITERS="$2"; shift 2 ;;
        --json)    OUT="$2"; shift 2 ;;
        --acc-env) ACC_ENV="$2"; shift 2 ;;
        --no-build) BUILD=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

[ -n "$JEFF" ] || { echo "error: --jeff <dir> is required" >&2; exit 2; }
[ -n "$LAYA" ] || { echo "error: --laya <dir> is required" >&2; exit 2; }
[ -d "$JEFF" ] || { echo "error: no such directory: $JEFF" >&2; exit 2; }
[ -d "$LAYA" ] || { echo "error: no such directory: $LAYA" >&2; exit 2; }

if [ "$BUILD" = 1 ]; then
    echo ">>> cargo build --release -p jeff-infer -p laya-infer --examples" >&2
    cargo build --release -p jeff-infer -p laya-infer --examples >&2
fi

if [ -z "$OUT" ]; then
    OUT="$(mktemp -d)"
    trap 'rm -rf "$OUT"' EXIT
fi
mkdir -p "$OUT"

run() {
    echo ">>> $*" >&2
    "$@"
}

# Rust: one shared thread budget via RAYON_NUM_THREADS.
run env RAYON_NUM_THREADS="$THREADS" \
    ./target/release/examples/bench_jeff "$JEFF" "$WARMUP" "$ITERS" > "$OUT/rust_jeff.json"
run env RAYON_NUM_THREADS="$THREADS" \
    ./target/release/examples/bench_laya "$LAYA" "$WARMUP" "$ITERS" > "$OUT/rust_laya.json"

# Julia: -t N; the scripts set BLAS threads through the package's CPU policy.
run julia "-t$THREADS" --project=extern/JeffClient.jl \
    tools/bench_jeff_real.jl "$JEFF" "$WARMUP" "$ITERS" > "$OUT/julia_jeff.json"
run julia "-t$THREADS" --project=extern/Laya.jl \
    tools/bench_laya_real.jl "$LAYA" "$WARMUP" "$ITERS" > "$OUT/julia_laya.json"

# Optional Apple-silicon runs with Accelerate BLAS (Julia's fastest CPU path).
# The same environment provides AppleAccelerate for both models.
if [ -n "$ACC_ENV" ]; then
    if [ "$(uname -m)" != "arm64" ]; then
        echo "warning: --acc-env is Apple-silicon only; skipping" >&2
    elif [ ! -d "$ACC_ENV" ]; then
        echo "error: no such --acc-env directory: $ACC_ENV" >&2
        exit 2
    else
        run julia "-t$THREADS" --project="$ACC_ENV" \
            tools/bench_jeff_real_accelerate.jl "$JEFF" "$WARMUP" "$ITERS" \
            > "$OUT/julia_jeff_accelerate.json"
        run julia "-t$THREADS" --project="$ACC_ENV" \
            tools/bench_laya_real_accelerate.jl "$LAYA" "$WARMUP" "$ITERS" \
            > "$OUT/julia_laya_accelerate.json"
    fi
fi

python3 "$ROOT/tools/bench_table.py" "$OUT" "$THREADS" "$WARMUP" "$ITERS"
