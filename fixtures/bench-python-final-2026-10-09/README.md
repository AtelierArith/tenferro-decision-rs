# Original Python CPU comparison

Measurements use F32, eight threads, five warm-ups and 15 timed forwards.
Loading and correctness checks are outside timed regions. Run each benchmark
sequentially on the same machine. `Laya` uses `laya-infer/onednn`; `Jeff` uses
the default `HostOpt` engine implementation with its activation workspace.

Set `LAYA_SNAPSHOT`, `JEFF_SNAPSHOT`, `LAYA_PYTHON_SOURCE`, and `PYTHON_BIN` to
your checkpoint snapshots, clean Laya upstream checkout, and benchmark Python
environment. Install oneDNN 3.1.1 headers/library (or supply
`ONEDNN_INCLUDE_DIR` / `ONEDNN_LIB_DIR`) and make its shared library visible to
the runtime loader. Python environment/package and upstream source versions are
recorded in the JSON outputs. Checkpoints are not included here.

```sh
cargo build --release -p laya-infer --features onednn --example bench_laya_tenferro_gap
OMP_NUM_THREADS=8 RAYON_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE target/release/examples/bench_laya_tenferro_gap "$LAYA_SNAPSHOT" 5 15
env -u OMP_WAIT_POLICY OMP_NUM_THREADS=8 "$PYTHON_BIN" tools/bench_python_real.py laya "$LAYA_SNAPSHOT" --source "$LAYA_PYTHON_SOURCE" --threads 8 --warmup 5 --iters 15
cargo build --release -p jeff-infer --example bench_jeff
OMP_NUM_THREADS=8 RAYON_NUM_THREADS=8 target/release/examples/bench_jeff "$JEFF_SNAPSHOT" 5 15
env -u OMP_WAIT_POLICY OMP_NUM_THREADS=8 "$PYTHON_BIN" tools/bench_python_real.py jeff "$JEFF_SNAPSHOT" --source extern/JeffClient.jl/extern/jeff --threads 8 --warmup 5 --iters 15
```

`jeff-rust.json` omits the machine-specific checkpoint path from the original
output; timings and logits are unchanged. Laya's raw results and two-run
comparison are in the adjacent `bench-onednn-owned-2026-10-09` directory.
Rust metadata identifies the parent revision; measurements include the
owned-provider worktree committed as the `implementation_revision` in
[`validation.json`](validation.json). That record identifies the implementation,
comparison evidence, required checks and remaining scope limitations.
