//! Benchmark the Rust Jeff CPU forward on a production checkpoint.
//!
//!     cargo run --release -p jeff-infer --example bench_jeff -- <CHECKPOINT_DIR> [WARMUP ITERS]
//!
//! Uses the same fixed prepared inputs as `tools/bench_jeff_real.jl` and
//! reports model loading separately from the forward. The tenferro path is
//! timed on the shortest sequence only.

use std::time::Instant;

use jeff_infer::checkpoint::load_checkpoint;
use jeff_infer::host_opt::{HostOptWorkspace, forward_host_opt_with};
use jeff_infer::model::{
    DeltaKernel, forward_reference, forward_tenferro, forward_tenferro_cached_kernel,
};
use serde_json::json;
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;
use tenferro_gated_delta::GatedDeltaWorkspace;
use tenferro_infer::TensorCache;

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
    let mut host_opt_ws = HostOptWorkspace::new();
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

        // The host-optimized path with a reused activation workspace.
        for _ in 0..warmup {
            let _ = forward_host_opt_with(&mut host_opt_ws, &cfg, &weights, &ids, &mask).unwrap();
        }
        let mut opt_samples = Vec::new();
        for _ in 0..iters {
            let timer = Instant::now();
            let _ = forward_host_opt_with(&mut host_opt_ws, &cfg, &weights, &ids, &mask).unwrap();
            opt_samples.push(timer.elapsed().as_secs_f64() * 1000.0);
        }
        value["host_opt"] = stats(opt_samples);
        shapes.push(value);
    }

    // The tenferro path: fresh weight cache (rebuilds weights) vs a reused
    // weight cache + DeltaNet workspace. Timed on the shortest sequence.
    let length = 8usize;
    let ids: Vec<i64> = (0..length).map(|i| BASE[i % BASE.len()]).collect();
    let mask = vec![1.0f32; length];
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();

    let time = |cached: bool, kernel: DeltaKernel| -> serde_json::Value {
        let mut workspace = GatedDeltaWorkspace::new();
        let mut cache = TensorCache::new();
        let run = |workspace: &mut GatedDeltaWorkspace, cache: &mut TensorCache| -> f64 {
            let timer = Instant::now();
            let _ = runtime
                .with_eager_session(|session| {
                    if cached {
                        forward_tenferro_cached_kernel(
                            workspace, cache, session, &cfg, &weights, &ids, &mask, kernel,
                        )
                    } else {
                        forward_tenferro(session, &cfg, &weights, &ids, &mask)
                    }
                })
                .unwrap()
                .unwrap();
            timer.elapsed().as_secs_f64() * 1000.0
        };
        let _ = run(&mut workspace, &mut cache);
        let mut samples = Vec::new();
        for _ in 0..2 {
            samples.push(run(&mut workspace, &mut cache));
        }
        let mut value = stats(samples);
        value["cached"] = json!(cached);
        value["delta_kernel"] = json!(format!("{kernel:?}"));
        value["cache_entries"] = json!(cache.len());
        value
    };
    let tenferro = time(false, DeltaKernel::default());
    let tenferro_cached = time(true, DeltaKernel::HostRecurrent);
    let tenferro_native = time(true, DeltaKernel::TensorNative);

    let mut metadata = bench_suite::BenchMetadata::capture();
    metadata.profile = Some(
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
        .into(),
    );
    let metadata: serde_json::Value = serde_json::from_str(&metadata.to_json()).unwrap();
    let out = json!({
        "runtime": "rust-cpu",
        "metadata": metadata,
        "rayon_threads": rayon::current_num_threads(),
        "checkpoint": dir,
        "load_ms": load_ms,
        "warmup": warmup,
        "shapes": shapes,
        "tenferro_forward_8": tenferro,
        "tenferro_cached_forward_8": tenferro_cached,
        "tenferro_native_forward_8": tenferro_native,
    });
    println!("{}", serde_json::to_string(&out).unwrap());
}
