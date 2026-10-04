//! Benchmark the Rust Jeff CPU forward on a production checkpoint.
//!
//!     cargo run --release -p jeff-infer --example bench_jeff -- <CHECKPOINT_DIR> [WARMUP ITERS]
//!
//! Uses the same fixed prepared inputs as `tools/bench_jeff_real.jl` and
//! reports model loading separately from the forward. The tenferro path is
//! timed on the shortest sequence only.

use std::time::Instant;

use jeff_infer::checkpoint::load_checkpoint;
use jeff_infer::model::{forward_reference, forward_tenferro};
use serde_json::json;
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;

const BASE: [i64; 8] = [2, 100, 1000, 2000, 3000, 4000, 5, 3];

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
        .expect("usage: bench_jeff <checkpoint_dir> [warmup iters]");
    let warmup: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(2);
    let iters: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(5);

    let start = Instant::now();
    let checkpoint = load_checkpoint(dir).expect("load checkpoint");
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let (cfg, weights) = (checkpoint.config, checkpoint.weights);

    let mut shapes = Vec::new();
    for length in [8usize, 16, 64] {
        let ids: Vec<i64> = (0..length).map(|i| BASE[i % BASE.len()]).collect();
        let mask = vec![1.0f32; length];
        for _ in 0..warmup {
            let _ = forward_reference(&cfg, &weights, &ids, &mask).unwrap();
        }
        let mut samples = Vec::new();
        for _ in 0..iters {
            let timer = Instant::now();
            let _ = forward_reference(&cfg, &weights, &ids, &mask).unwrap();
            samples.push(timer.elapsed().as_secs_f64() * 1000.0);
        }
        let mut value = stats(samples);
        value["length"] = json!(length);
        value["batch"] = json!(1);
        shapes.push(value);
    }

    // The tenferro path rebuilds every weight tensor per call; time it once on
    // the shortest sequence.
    let length = 8usize;
    let ids: Vec<i64> = (0..length).map(|i| BASE[i % BASE.len()]).collect();
    let mask = vec![1.0f32; length];
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let _ = runtime
        .with_eager_session(|session| forward_tenferro(session, &cfg, &weights, &ids, &mask))
        .unwrap()
        .unwrap();
    let mut tenferro_samples = Vec::new();
    for _ in 0..2 {
        let timer = Instant::now();
        let _ = runtime
            .with_eager_session(|session| forward_tenferro(session, &cfg, &weights, &ids, &mask))
            .unwrap()
            .unwrap();
        tenferro_samples.push(timer.elapsed().as_secs_f64() * 1000.0);
    }
    let mut tenferro = stats(tenferro_samples);
    tenferro["length"] = json!(length);
    tenferro["batch"] = json!(1);

    let out = json!({
        "runtime": "rust-cpu",
        "checkpoint": dir,
        "load_ms": load_ms,
        "warmup": warmup,
        "shapes": shapes,
        "tenferro_forward_8": tenferro,
    });
    println!("{}", serde_json::to_string(&out).unwrap());
}
