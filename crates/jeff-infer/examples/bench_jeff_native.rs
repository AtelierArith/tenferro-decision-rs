//! Cached tensor-native Jeff timings, with raw samples for each length.
//!
//! RAYON_NUM_THREADS=8 cargo run --release -p jeff-infer \
//!     --example bench_jeff_native -- <checkpoint_dir> [warmup iters]

use std::time::Instant;

use jeff_infer::checkpoint::load_checkpoint;
use jeff_infer::model::{DeltaKernel, forward_tenferro_cached_kernel};
use serde_json::json;
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;
use tenferro_gated_delta::GatedDeltaWorkspace;
use tenferro_infer::TensorCache;

const BASE: [i64; 8] = [2, 100, 1000, 2000, 3000, 4000, 5, 3];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .expect("usage: bench_jeff_native <checkpoint_dir> [warmup iters]");
    let warmup: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(2);
    let iters: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(5);
    assert!(iters > 0, "at least one measured iteration required");
    let checkpoint = load_checkpoint(dir).expect("load checkpoint");
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let mut workspace = GatedDeltaWorkspace::new();
    let mut cache = TensorCache::new();
    let mut rows = Vec::new();
    for length in [8usize, 16, 64] {
        let ids: Vec<i64> = (0..length).map(|i| BASE[i % BASE.len()]).collect();
        let mask = vec![1.0f32; length];
        let mut samples = Vec::new();
        let mut logits = Vec::new();
        for iteration in 0..warmup + iters {
            let timer = Instant::now();
            logits = runtime
                .with_eager_session(|session| {
                    forward_tenferro_cached_kernel(
                        &mut workspace,
                        &mut cache,
                        session,
                        &checkpoint.config,
                        &checkpoint.weights,
                        &ids,
                        &mask,
                        DeltaKernel::TensorNative,
                    )
                })
                .unwrap()
                .unwrap();
            if iteration >= warmup {
                samples.push(timer.elapsed().as_secs_f64() * 1000.0);
            }
        }
        let mut sorted = samples.clone();
        sorted.sort_by(f64::total_cmp);
        let median = if iters % 2 == 0 {
            (sorted[iters / 2 - 1] + sorted[iters / 2]) * 0.5
        } else {
            sorted[iters / 2]
        };
        rows.push(json!({"length": length, "batch": 1, "ms_median": median,
            "samples_ms": samples, "logits": logits}));
        eprintln!("native L{length}: {median:.3} ms");
    }
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
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "warmup": warmup, "iters": iters, "rows": rows, "metadata": metadata,
            "rayon_threads": rayon::current_num_threads(), "dtype": "f32",
            "delta_kernel": "TensorNative", "cached_weights": true,
            "cached_native_constants": true
        }))
        .unwrap()
    );
}
