# Current Julia target audit

Computation revision: d190d56417a0d6a95aee3486b53121f96fc7d813.
Julia scripts add raw samples and machine/BLAS metadata without changing model
computation. Checkpoint fields contain snapshot revisions instead of local paths.
Run sequentially with five warmups, 15 samples and eight threads:

```sh
julia -t8 --project=extern/Laya.jl tools/bench_laya_real.jl "$LAYA_SNAPSHOT" 5 15
julia -t8 --project=extern/JeffClient.jl tools/bench_jeff_real.jl "$JEFF_SNAPSHOT" 5 15
cargo build --release -p laya-infer --features onednn --example bench_laya_tenferro_gap
OMP_NUM_THREADS=8 RAYON_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE target/release/examples/bench_laya_tenferro_gap "$LAYA_SNAPSHOT" 5 15
```

oneDNN 3.1.1 must be visible to the runtime loader. Julia uses its package's
normal CPU policy; reported BLAS settings are authoritative. This is one run
per runtime, so small timing differences require repetition. It proves that
the existing Python comparison cannot substitute for Julia target validation.
