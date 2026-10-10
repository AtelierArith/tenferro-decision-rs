//! Warm Laya tenferro forward latency on a CUDA device vs the CPU.
//!
//!     CUDA_VISIBLE_DEVICES=1 cargo run --release -p laya-infer --features cuda \
//!         --example bench_laya_cuda -- <CHECKPOINT_DIR> [WARMUP ITERS] [--cpu]
//!
//! Uses the fixed prepared batches of `bench_laya` / `tools/bench_laya_real.jl`
//! (L8B1, L64B1, L8B8). Each timed call is one `forward_tenferro_cached` with a
//! reused weight cache: it includes uploading ids/masks, the forward, the
//! intermediate marker-logit/CLS readouts the model pools on the host, the
//! final downloads and the synchronization they imply. Checkpoint loading and
//! the first (weight-upload) call are excluded. `--cpu` also times the CPU
//! tenferro forward on the same inputs.

use std::sync::Arc;
use std::time::Instant;

use laya_infer::agent::Device;
use laya_infer::checkpoint::{LayaCheckpoint, load_checkpoint};
use laya_infer::model::{TensorCache, forward_tenferro_cached};
use serde_json::json;
use tenferro_ad::EagerRuntime;

const BASE: [i64; 8] = [2, 100, 1000, 2000, 3000, 4000, 5, 3];

type Batch = (Vec<i64>, Vec<bool>, Vec<i64>, Vec<bool>, Vec<i64>);

fn make_batch(length: usize, batch: usize) -> Batch {
    let mut ids = Vec::with_capacity(length * batch);
    for _ in 0..batch {
        for l in 0..length {
            ids.push(BASE[l % BASE.len()]);
        }
    }
    let mut marker_pos = Vec::with_capacity(3 * batch);
    for _ in 0..batch {
        marker_pos.extend_from_slice(&[1, 3, 5]);
    }
    (
        ids,
        vec![true; length * batch],
        marker_pos,
        vec![true; 3 * batch],
        vec![0i64; batch],
    )
}

fn stats(mut samples: Vec<f64>) -> serde_json::Value {
    let raw = samples.clone();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len();
    let median = if n % 2 == 1 {
        samples[n / 2]
    } else {
        0.5 * (samples[n / 2 - 1] + samples[n / 2])
    };
    json!({
        "ms_median": median,
        "ms_min": samples[0],
        "ms_max": samples[n - 1],
        "iterations": n,
        "samples_ms": raw,
    })
}

fn time(
    runtime: &Arc<EagerRuntime>,
    cache: &mut TensorCache,
    checkpoint: &LayaCheckpoint,
    batch: &Batch,
    warmup: usize,
    iters: usize,
) -> (serde_json::Value, f64) {
    let (ids, mask, marker_pos, marker_mask, qtype) = batch;
    let mut run = || {
        runtime
            .with_eager_session(|session| {
                forward_tenferro_cached(
                    session,
                    cache,
                    &checkpoint.encoder,
                    &checkpoint.agent,
                    &checkpoint.weights,
                    ids,
                    mask,
                    marker_pos,
                    marker_mask,
                    qtype,
                )
            })
            .unwrap()
            .unwrap()
    };
    let first = Instant::now();
    run();
    let first_ms = first.elapsed().as_secs_f64() * 1e3;
    for _ in 0..warmup {
        run();
    }
    let samples = (0..iters)
        .map(|_| {
            let timer = Instant::now();
            std::hint::black_box(run());
            timer.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    (stats(samples), first_ms)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .expect("usage: bench_laya_cuda <checkpoint_dir> [warmup iters] [--cpu]");
    let warmup: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(30);
    let with_cpu = args.iter().any(|arg| arg == "--cpu");

    let start = Instant::now();
    let checkpoint = load_checkpoint(dir).expect("load checkpoint");
    let load_ms = start.elapsed().as_secs_f64() * 1e3;
    let gpu = Device::Cuda(0).runtime().expect("CUDA device");
    let cpu = Device::Cpu.runtime().unwrap();
    let mut gpu_cache = TensorCache::new();
    let mut cpu_cache = TensorCache::new();

    // `LAYA_BENCH_SHAPES=8x1,64x1,93x10` (LENGTHxBATCH) overrides the default set.
    let shape_list: Vec<(usize, usize)> = std::env::var("LAYA_BENCH_SHAPES")
        .ok()
        .map(|spec| {
            spec.split(',')
                .map(|item| {
                    let (l, b) = item.split_once('x').expect("LENGTHxBATCH");
                    (l.trim().parse().unwrap(), b.trim().parse().unwrap())
                })
                .collect()
        })
        .unwrap_or_else(|| vec![(8, 1), (64, 1), (8, 8)]);
    let mut shapes = Vec::new();
    for (length, batch) in shape_list {
        let inputs = make_batch(length, batch);
        let (cuda, first_ms) = time(&gpu, &mut gpu_cache, &checkpoint, &inputs, warmup, iters);
        let mut shape = json!({
            "length": length,
            "batch": batch,
            "cuda": cuda,
            "cuda_first_call_ms": first_ms,
        });
        if with_cpu {
            shape["cpu_tenferro"] = time(
                &cpu,
                &mut cpu_cache,
                &checkpoint,
                &inputs,
                warmup.min(3),
                iters.min(10),
            )
            .0;
        }
        eprintln!(
            "L{length}B{batch}: cuda median {:.2} ms (first {:.0} ms)",
            shape["cuda"]["ms_median"].as_f64().unwrap(),
            first_ms
        );
        shapes.push(shape);
    }
    let out = json!({
        "runtime": "rust-cuda",
        "checkpoint": dir,
        "load_ms": load_ms,
        "warmup": warmup,
        "shapes": shapes,
    });
    println!("{}", serde_json::to_string(&out).unwrap());
}
