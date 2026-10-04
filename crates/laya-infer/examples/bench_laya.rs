//! Benchmark the Rust Laya CPU forward on a production checkpoint.
//!
//!     cargo run --release -p laya-infer --example bench_laya -- <CHECKPOINT_DIR> [WARMUP ITERS]
//!
//! Uses the same fixed prepared batches as `tools/bench_laya_real.jl` and
//! reports model loading separately from the forward. The tenferro path is
//! timed once on the smallest shape (it rebuilds tensors per call).

use std::time::Instant;

use laya_infer::checkpoint::load_checkpoint;
use laya_infer::model::{
    forward_reference, forward_tenferro, forward_tenferro_cached, TensorCache,
};
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

fn stats(mut samples: Vec<f64>) -> serde_json::Value {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len();
    let median = if n % 2 == 1 {
        samples[n / 2]
    } else {
        0.5 * (samples[n / 2 - 1] + samples[n / 2])
    };
    json!({
        "ms_median": median,
        "ms_min": samples.first().copied().unwrap_or(0.0),
        "ms_mean": samples.iter().sum::<f64>() / n as f64,
        "iterations": n,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .expect("usage: bench_laya <checkpoint_dir> [warmup iters]");
    let warmup: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(2);
    let iters: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(5);

    let start = Instant::now();
    let checkpoint = load_checkpoint(dir).expect("load checkpoint");
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;

    let mut shapes = Vec::new();
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
        let mut samples = Vec::new();
        for _ in 0..iters {
            let timer = Instant::now();
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
            samples.push(timer.elapsed().as_secs_f64() * 1000.0);
        }
        let mut value = stats(samples);
        value["length"] = json!(length);
        value["batch"] = json!(batch);
        shapes.push(value);
    }

    // The tenferro path: rebuilding every weight tensor per call (fresh cache)
    // vs a reused weight cache. Timed on the smallest shape.
    let (ids, mask, marker_pos, marker_mask, qtype) = make_batch(8, 1);
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();

    let time = |cached: bool| -> serde_json::Value {
        let mut cache = TensorCache::new();
        let _ = runtime
            .with_eager_session(|session| {
                if cached {
                    forward_tenferro_cached(
                        session,
                        &mut cache,
                        &checkpoint.encoder,
                        &checkpoint.agent,
                        &checkpoint.weights,
                        &ids,
                        &mask,
                        &marker_pos,
                        &marker_mask,
                        &qtype,
                    )
                } else {
                    forward_tenferro(
                        session,
                        &checkpoint.encoder,
                        &checkpoint.agent,
                        &checkpoint.weights,
                        &ids,
                        &mask,
                        &marker_pos,
                        &marker_mask,
                        &qtype,
                    )
                }
            })
            .unwrap()
            .unwrap();
        let mut samples = Vec::new();
        for _ in 0..2 {
            let timer = Instant::now();
            let _ = runtime
                .with_eager_session(|session| {
                    if cached {
                        forward_tenferro_cached(
                            session,
                            &mut cache,
                            &checkpoint.encoder,
                            &checkpoint.agent,
                            &checkpoint.weights,
                            &ids,
                            &mask,
                            &marker_pos,
                            &marker_mask,
                            &qtype,
                        )
                    } else {
                        forward_tenferro(
                            session,
                            &checkpoint.encoder,
                            &checkpoint.agent,
                            &checkpoint.weights,
                            &ids,
                            &mask,
                            &marker_pos,
                            &marker_mask,
                            &qtype,
                        )
                    }
                })
                .unwrap()
                .unwrap();
            samples.push(timer.elapsed().as_secs_f64() * 1000.0);
        }
        let mut value = stats(samples);
        value["cached"] = json!(cached);
        value["cache_entries"] = json!(cache.len());
        value
    };
    let tenferro = time(false);
    let tenferro_cached = time(true);

    let out = json!({
        "runtime": "rust-cpu",
        "checkpoint": dir,
        "load_ms": load_ms,
        "warmup": warmup,
        "shapes": shapes,
        "tenferro_forward_8x1": tenferro,
        "tenferro_cached_forward_8x1": tenferro_cached,
    });
    println!("{}", serde_json::to_string(&out).unwrap());
}
