//! Per-shape gap between the cached tenferro forward and `host_opt` for Jeff.
//!
//! The shipped `bench_jeff` example times the tenferro path only at the
//! shortest sequence; this times every shape.
//!
//!     RAYON_NUM_THREADS=8 cargo run --release -p jeff-infer \
//!         --example bench_jeff_tenferro_gap -- <dir> [warmup iters]

use std::time::Instant;

use jeff_infer::checkpoint::load_checkpoint;
use jeff_infer::host_opt::{HostOptWorkspace, forward_host_opt_with};
use jeff_infer::model::{DeltaKernel, forward_tenferro_cached_kernel};
use serde_json::json;
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;
use tenferro_gated_delta::GatedDeltaWorkspace;
use tenferro_infer::TensorCache;

const BASE: [i64; 8] = [2, 100, 1000, 2000, 3000, 4000, 5, 3];

fn median(mut s: Vec<f64>) -> f64 {
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .expect("usage: bench_tenferro_gap <dir> [warmup iters]");
    let warmup: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(3);
    let iters: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(15);

    let checkpoint = load_checkpoint(dir).expect("load checkpoint");
    let (cfg, weights) = (checkpoint.config, checkpoint.weights);

    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let mut cache = TensorCache::new();
    let mut ws = GatedDeltaWorkspace::new();
    let mut host_ws = HostOptWorkspace::new();

    let mut rows = Vec::new();
    for length in [8usize, 16, 64] {
        let ids: Vec<i64> = (0..length).map(|i| BASE[i % BASE.len()]).collect();
        let mask = vec![1.0f32; length];

        for _ in 0..warmup {
            let _ = forward_host_opt_with(&mut host_ws, &cfg, &weights, &ids, &mask).unwrap();
        }
        let mut hs = Vec::new();
        for _ in 0..iters {
            let t = Instant::now();
            let _ = forward_host_opt_with(&mut host_ws, &cfg, &weights, &ids, &mask).unwrap();
            hs.push(t.elapsed().as_secs_f64() * 1000.0);
        }

        let run_te = |cache: &mut TensorCache, ws: &mut GatedDeltaWorkspace| -> f64 {
            let t = Instant::now();
            let _ = runtime
                .with_eager_session(|session| {
                    forward_tenferro_cached_kernel(
                        ws,
                        cache,
                        session,
                        &cfg,
                        &weights,
                        &ids,
                        &mask,
                        DeltaKernel::HostRecurrent,
                    )
                })
                .unwrap()
                .unwrap();
            t.elapsed().as_secs_f64() * 1000.0
        };
        for _ in 0..warmup {
            let _ = run_te(&mut cache, &mut ws);
        }
        let mut ts = Vec::new();
        for _ in 0..iters {
            ts.push(run_te(&mut cache, &mut ws));
        }

        rows.push(json!({
            "length": length,
            "host_opt_ms": median(hs),
            "tenferro_ms": median(ts),
        }));
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "warmup": warmup, "iters": iters, "rows": rows,
        }))
        .unwrap()
    );
}
