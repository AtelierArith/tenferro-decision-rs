//! Per-shape gap between the cached tenferro forward and the host
//! (`forward_reference`) for Laya.
//!
//! The shipped `bench_laya` example times the tenferro path only at L8 B1; this
//! times every benchmark shape.
//!
//!     RAYON_NUM_THREADS=8 cargo run --release -p laya-infer \
//!         --example bench_laya_tenferro_gap -- <dir> [warmup iters]

use std::time::Instant;

use laya_infer::checkpoint::load_checkpoint;
use laya_infer::model::{TensorCache, forward_reference, forward_tenferro_cached};
use serde_json::json;
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;

const BASE: [i64; 8] = [2, 100, 1000, 2000, 3000, 4000, 5, 3];

type Batch = (Vec<i64>, Vec<bool>, Vec<i64>, Vec<bool>, Vec<i64>);

fn make_batch(length: usize, batch: usize) -> Batch {
    let mut ids = Vec::with_capacity(length * batch);
    for _ in 0..batch {
        for l in 0..length {
            ids.push(BASE[l % BASE.len()]);
        }
    }
    let mask = vec![true; length * batch];
    let mut marker_pos = Vec::with_capacity(3 * batch);
    for _ in 0..batch {
        marker_pos.extend_from_slice(&[1, 3, 5]);
    }
    let marker_mask = vec![true; 3 * batch];
    let qtype = vec![0i64; batch];
    (ids, mask, marker_pos, marker_mask, qtype)
}

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
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let mut cache = TensorCache::new();

    let mut rows = Vec::new();
    for (length, batch) in [(8usize, 1usize), (64, 1), (8, 8)] {
        let (ids, mask, marker_pos, marker_mask, qtype) = make_batch(length, batch);

        for _ in 0..warmup {
            let _ = forward_reference(
                &checkpoint.encoder,
                &checkpoint.agent,
                &checkpoint.weights,
                &ids,
                &mask,
                &marker_pos,
                &marker_mask,
                &qtype,
            )
            .unwrap();
        }
        let mut hs = Vec::new();
        for _ in 0..iters {
            let t = Instant::now();
            let _ = forward_reference(
                &checkpoint.encoder,
                &checkpoint.agent,
                &checkpoint.weights,
                &ids,
                &mask,
                &marker_pos,
                &marker_mask,
                &qtype,
            )
            .unwrap();
            hs.push(t.elapsed().as_secs_f64() * 1000.0);
        }

        let run_te = |cache: &mut TensorCache| -> f64 {
            let t = Instant::now();
            let _ = runtime
                .with_eager_session(|session| {
                    forward_tenferro_cached(
                        session,
                        cache,
                        &checkpoint.encoder,
                        &checkpoint.agent,
                        &checkpoint.weights,
                        &ids,
                        &mask,
                        &marker_pos,
                        &marker_mask,
                        &qtype,
                    )
                })
                .unwrap()
                .unwrap();
            t.elapsed().as_secs_f64() * 1000.0
        };
        for _ in 0..warmup {
            let _ = run_te(&mut cache);
        }
        let mut ts = Vec::new();
        for _ in 0..iters {
            ts.push(run_te(&mut cache));
        }

        rows.push(json!({
            "length": length, "batch": batch,
            "host_ms": median(hs),
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
