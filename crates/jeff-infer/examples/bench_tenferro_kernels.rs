//! Compare the tenferro Jeff forward's DeltaNet kernels at several lengths.
//!
//!     RAYON_NUM_THREADS=8 cargo run --release -q -p jeff-infer --example bench_tenferro_kernels -- <CHECKPOINT_DIR> [ITERS]

use std::time::Instant;

use jeff_infer::checkpoint::load_checkpoint;
use jeff_infer::model::{DeltaKernel, forward_reference, forward_tenferro_cached_kernel};
use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;
use tenferro_gated_delta::GatedDeltaWorkspace;
use tenferro_infer::TensorCache;

const BASE: [i64; 8] = [2, 100, 1000, 2000, 3000, 4000, 5, 3];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).expect("usage: <dir> [iters]");
    let iters: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(5);
    let checkpoint = load_checkpoint(dir).expect("load checkpoint");
    let (cfg, weights) = (checkpoint.config, checkpoint.weights);
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();

    for length in [8usize, 16, 64] {
        let ids: Vec<i64> = (0..length).map(|i| BASE[i % BASE.len()]).collect();
        let mask = vec![1.0f32; length];

        let host = {
            let _ = forward_reference(&cfg, &weights, &ids, &mask).unwrap();
            let mut samples = Vec::new();
            for _ in 0..iters {
                let timer = Instant::now();
                let _ = forward_reference(&cfg, &weights, &ids, &mask).unwrap();
                samples.push(timer.elapsed().as_secs_f64() * 1e3);
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            samples[0]
        };

        let mut kernels = Vec::new();
        for kernel in [DeltaKernel::HostRecurrent, DeltaKernel::TensorNative] {
            let mut workspace = GatedDeltaWorkspace::new();
            let mut cache = TensorCache::new();
            let mut run = || {
                runtime
                    .with_eager_session(|session| {
                        forward_tenferro_cached_kernel(
                            &mut workspace,
                            &mut cache,
                            session,
                            &cfg,
                            &weights,
                            &ids,
                            &mask,
                            kernel,
                        )
                    })
                    .unwrap()
                    .unwrap()
            };
            let _ = run();
            let mut samples = Vec::new();
            for _ in 0..iters {
                let timer = Instant::now();
                let _ = run();
                samples.push(timer.elapsed().as_secs_f64() * 1e3);
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            kernels.push((kernel, samples[0]));
        }

        println!("len={length:3} best-of-{iters}: host {host:8.3} ms");
        for (kernel, ms) in kernels {
            println!(
                "        tenferro {kernel:?} {ms:8.3} ms  ({:.2}x host)",
                ms / host
            );
        }
    }
}
