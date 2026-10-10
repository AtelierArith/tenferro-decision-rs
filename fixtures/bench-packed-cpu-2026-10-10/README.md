# Portable production CPU comparison

Ryzen 9 PRO 8945HS, Linux x86_64, F32, eight runtime threads, five warmups,
15 timed forwards per shape, two independent runs per runtime. Model loading
and lazy weight/plan preparation are outside the forward timing. Rust uses the
**default build, without oneDNN**: Laya's cached tenferro path and Jeff's prepared
CPU path selected by Auto/HostOpt. The Jeff report also retains its oracle and
portable HostOpt diagnostic timings; compare Julia with `prepared_cpu`.

These runs used the working tree atop d190d56417a0d6a95aee3486b53121f96fc7d813.
`source-digests.json` identifies the measured Rust/harness source and Julia
submodule pins. Rust's original `metadata.git_rev` is deliberately unchanged:
it records that parent commit, since the measurements preceded committing the
implementation. Checkpoint fields contain snapshot revisions, not local paths.

Processes ran sequentially, in the order Rust Laya, Rust Jeff, Julia Laya,
Julia Jeff, then the same order again. `*.environment.json` samples other
process CPU time every 0.5 seconds. There were no competing Julia, Rust compiler,
or model jobs. The first Rust Laya run observed two brief Codex CPU bursts,
and the first Rust Jeff run observed one brief kache daemon CPU burst.
The remaining six runs observed no other process above 0.25 cores per interval.
These records do not guarantee zero background activity. Thermal, scheduling
and sequential-order drift remain: Laya L64 medians vary from 116.9 to 128.4 ms.
Report both runs; do not claim that Rust beats Julia at every shape.

| model / shape | Rust run 1 / 2 (ms) | Julia run 1 / 2 (ms) |
|---|---:|---:|
| Laya L8 B1 | 63.35 / 61.90 | 70.48 / 71.41 |
| Laya L16 B1 | 67.38 / 68.29 | 69.26 / 69.36 |
| Laya L64 B1 | 128.41 / 116.94 | 122.68 / 123.89 |
| Laya L8 B8 | 119.86 / 123.00 | 119.72 / 120.15 |
| Jeff L8 B1 | 59.10 / 58.92 | 75.85 / 83.28 |
| Jeff L16 B1 | 71.07 / 72.43 | 85.62 / 93.52 |
| Jeff L64 B1 | 166.15 / 168.57 | 163.24 / 159.59 |

The paired run ratios range from 0.71 to 1.06. Long-input/batch cases are near
parity, with up to 5.6% slower Rust medians; short Jeff forwards are 17–29%
faster. This closes the original multi-fold CPU gap, without establishing a
universal speed win or any GPU result.

Every Rust benchmark shape checks its output against the host oracle and
retains logits (and Laya actions). Maximum observed absolute error is
0.00244140625 for Laya's combined logits/actions and 0.000026226044 for Jeff;
the existing scale-aware tolerances pass. Both runs produce identical outputs.
The production Laya bundled questions/logits/actions, Jeff production reference
and engine answers, and Jeff Text/Json integration tests pass against the
committed Julia captures. Workspace tests, formatting and all-target Clippy
also pass. Both default release binaries link no oneDNN shared library.

Reproduce from the repository root, with cached snapshots resolved through
`hf-fetch`; set `LAYA_SNAPSHOT` and `JEFF_SNAPSHOT` to those local directories:

```sh
cargo build --release -p laya-infer -p jeff-infer \
    --example bench_laya_tenferro_gap --example bench_jeff
RAYON_NUM_THREADS=8 python3 fixtures/bench-packed-cpu-2026-10-10/monitor.py \
    /tmp/rust-laya.json target/release/examples/bench_laya_tenferro_gap \
    "$LAYA_SNAPSHOT" 5 15
RAYON_NUM_THREADS=8 python3 fixtures/bench-packed-cpu-2026-10-10/monitor.py \
    /tmp/rust-jeff.json target/release/examples/bench_jeff \
    "$JEFF_SNAPSHOT" 5 15 --host-only
python3 fixtures/bench-packed-cpu-2026-10-10/monitor.py /tmp/julia-laya.json \
    julia -t8 --project=extern/Laya.jl tools/bench_laya_real.jl "$LAYA_SNAPSHOT" 5 15
python3 fixtures/bench-packed-cpu-2026-10-10/monitor.py /tmp/julia-jeff.json \
    julia -t8 --project=extern/JeffClient.jl tools/bench_jeff_real.jl "$JEFF_SNAPSHOT" 5 15
# Repeat these four sequential commands for run 2.
```

The Linux monitor records process utilization, not model arithmetic. Julia uses
its normal package CPU policy; the reported BLAS settings are authoritative.
Interrupted/competing earlier runs in temporary files were excluded.
