//! Time one production Gated DeltaNet layer (the fused host recurrent kernel)
//! and the full host forward, to attribute the CPU cost.
//!
//!     RAYON_NUM_THREADS=8 cargo run --release -q -p jeff-infer --example delta_layer_time -- <CHECKPOINT_DIR> [ITERS]

use std::time::Instant;

use jeff_infer::checkpoint::load_checkpoint;
use jeff_infer::host_opt::{HostOptWorkspace, forward_host_opt_with};
use jeff_infer::model::AttentionWeights;
use tenferro_gated_delta::{GatedDeltaWorkspace, delta_layer_recurrent};

const BASE: [i64; 8] = [2, 100, 1000, 2000, 3000, 4000, 5, 3];

fn best_of<F: FnMut()>(mut f: F, warmup: usize, iters: usize) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let timer = Instant::now();
        f();
        best = best.min(timer.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).expect("usage: delta_layer_time <dir> [iters]");
    let iters: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(10);
    let checkpoint = load_checkpoint(dir).expect("load checkpoint");
    let (cfg, weights) = (checkpoint.config, checkpoint.weights);

    // Layer 0 is a Gated DeltaNet layer.
    let (delta_weights, delta_config) = match &weights.layers[0].attention {
        AttentionWeights::Delta { weights, config } => (weights, config),
        _ => panic!("layer 0 is not a delta layer"),
    };
    let delta_layers = weights
        .layers
        .iter()
        .filter(|l| matches!(l.attention, AttentionWeights::Delta { .. }))
        .count();

    for length in [8usize, 16, 64] {
        let ids: Vec<i64> = (0..length).map(|i| BASE[i % BASE.len()]).collect();
        let mask = vec![1.0f32; length];
        // A representative layer input.
        let x: Vec<f32> = (0..cfg.hidden * length)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
            .collect();

        let mut ws = GatedDeltaWorkspace::new();
        let delta_ms = best_of(
            || {
                let _ =
                    delta_layer_recurrent(delta_config, delta_weights, &x, &mask, &mut ws).unwrap();
            },
            2,
            iters,
        );

        let mut host = HostOptWorkspace::new();
        let full_ms = best_of(
            || {
                let _ = forward_host_opt_with(&mut host, &cfg, &weights, &ids, &mask).unwrap();
            },
            2,
            iters,
        );

        println!(
            "len={length:3}: one delta layer {delta_ms:7.2} ms x {delta_layers} = {:7.1} ms   full forward {full_ms:7.1} ms",
            delta_ms * delta_layers as f64
        );
    }
}
