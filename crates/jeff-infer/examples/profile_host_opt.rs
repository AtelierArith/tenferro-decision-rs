//! Run the host-optimized Jeff forward repeatedly, for sampling profilers.
//!
//!     RAYON_NUM_THREADS=8 cargo run --release -q -p jeff-infer --example profile_host_opt -- <CHECKPOINT_DIR> [ITERS] [LENGTH]

use std::time::Instant;

use jeff_infer::checkpoint::load_checkpoint;
use jeff_infer::host_opt::{HostOptWorkspace, forward_host_opt_with};

const BASE: [i64; 8] = [2, 100, 1000, 2000, 3000, 4000, 5, 3];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .expect("usage: profile_host_opt <dir> [iters] [length]");
    let iters: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(50);
    let length: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(64);
    let checkpoint = load_checkpoint(dir).expect("load checkpoint");
    let (cfg, weights) = (checkpoint.config, checkpoint.weights);
    let ids: Vec<i64> = (0..length).map(|i| BASE[i % BASE.len()]).collect();
    let mask = vec![1.0f32; length];
    let mut ws = HostOptWorkspace::new();
    for i in 0..iters {
        let t = Instant::now();
        let _ = forward_host_opt_with(&mut ws, &cfg, &weights, &ids, &mask).unwrap();
        eprintln!("iter {i}: {:.3} ms", t.elapsed().as_secs_f64() * 1e3);
    }
}
