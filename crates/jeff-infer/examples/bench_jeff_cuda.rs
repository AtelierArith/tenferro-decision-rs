//! Warm Jeff forward latency on a CUDA device vs the CPU paths.
//!
//!     CUDA_VISIBLE_DEVICES=1 cargo run --release -p jeff-infer --features cuda \
//!         --example bench_jeff_cuda -- <CHECKPOINT_DIR> [WARMUP ITERS] [--cpu]
//!
//! The input is JeffClient.jl's `examples/data/parcel_reference.json` case 1
//! (B1/L256, 101 active tokens after left padding; the engine trims the
//! padding, so the forward runs at L101), plus synthetic L8/L64 prompts.
//! Every timed call is `JeffEngine::logits`: it includes the token/mask
//! upload, the forward, the readout download and the device synchronization
//! the download implies. Checkpoint loading and the first (weight-upload,
//! NVRTC compile) call are excluded. `--cpu` also times the CPU `HostOpt`
//! and CPU tenferro engines on the same inputs. `parcel_full_L256` times the
//! model forward on all 256 positions (left padding kept, masked) with the
//! same retained device state, for comparison with full-sequence runtimes.

use std::time::Instant;

use decision_core::PreparedState;
use jeff_infer::engine::{Device, JeffBackend, JeffEngine};
use serde_json::json;

const BASE: [i64; 8] = [2, 100, 1000, 2000, 3000, 4000, 5, 3];

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
    engine: &mut JeffEngine,
    state: &PreparedState,
    warmup: usize,
    iters: usize,
) -> (serde_json::Value, f64) {
    let first = Instant::now();
    engine.logits(state).unwrap();
    let first_ms = first.elapsed().as_secs_f64() * 1e3;
    for _ in 0..warmup {
        engine.logits(state).unwrap();
    }
    let samples = (0..iters)
        .map(|_| {
            let timer = Instant::now();
            std::hint::black_box(engine.logits(state).unwrap());
            timer.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    (stats(samples), first_ms)
}

fn parcel() -> PreparedState {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../extern/JeffClient.jl/examples/data/parcel_reference.json"
    );
    let value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("parcel reference")).unwrap();
    let inputs = &value[0]["inputs"];
    PreparedState {
        input_ids: vec![
            inputs["input_ids"][0]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect(),
        ],
        attention_mask: vec![
            inputs["attention_mask"][0]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() != 0)
                .collect(),
        ],
    }
}

fn synthetic(length: usize) -> PreparedState {
    PreparedState {
        input_ids: vec![(0..length).map(|i| BASE[i % BASE.len()]).collect()],
        attention_mask: vec![vec![true; length]],
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .expect("usage: bench_jeff_cuda <checkpoint_dir> [warmup iters] [--cpu]");
    let warmup: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(30);
    let with_cpu = args.iter().any(|arg| arg == "--cpu");

    let start = Instant::now();
    let mut gpu = JeffEngine::load_with_device(dir, Device::Cuda(0)).expect("load on CUDA");
    let load_ms = start.elapsed().as_secs_f64() * 1e3;
    let mut cpu = with_cpu.then(|| {
        let opt = JeffEngine::load(dir).unwrap();
        let checkpoint = jeff_infer::checkpoint::load_checkpoint(dir).unwrap();
        let tenferro = JeffEngine::with_backend(
            checkpoint.config,
            checkpoint.decision,
            checkpoint.weights,
            JeffBackend::Tenferro,
        )
        .unwrap();
        (opt, tenferro)
    });

    let mut cases = Vec::new();
    for (name, state) in [
        ("parcel_L256_active101", parcel()),
        ("synthetic_L8", synthetic(8)),
        ("synthetic_L64", synthetic(64)),
    ] {
        let (cuda, first_ms) = time(&mut gpu, &state, warmup, iters);
        let mut case = json!({
            "name": name,
            "cuda": cuda,
            "cuda_first_call_ms": first_ms,
        });
        if let Some((opt, tenferro)) = cpu.as_mut() {
            case["cpu_host_opt"] = time(opt, &state, warmup.min(3), iters.min(10)).0;
            case["cpu_tenferro"] = time(tenferro, &state, warmup.min(3), iters.min(10)).0;
        }
        eprintln!(
            "{name}: cuda median {:.2} ms (first {:.0} ms)",
            case["cuda"]["ms_median"].as_f64().unwrap(),
            first_ms
        );
        cases.push(case);
    }
    // Full-sequence parcel forward (no trimming), same device path.
    {
        use jeff_infer::model::{CudaDeltaWorkspaces, DeltaKernel, forward_tenferro_device};
        let state = parcel();
        let ids = state.input_ids[0].clone();
        let mask: Vec<f32> = state.attention_mask[0]
            .iter()
            .map(|active| if *active { 1.0 } else { 0.0 })
            .collect();
        let runtime = Device::Cuda(0).runtime().unwrap();
        let mut workspace = tenferro_gated_delta::GatedDeltaWorkspace::new();
        let mut cuda = CudaDeltaWorkspaces::new();
        let mut cache = tenferro_infer::TensorCache::new();
        let mut run = || {
            runtime
                .with_eager_session(|session| {
                    forward_tenferro_device(
                        &mut workspace,
                        &mut cuda,
                        &mut cache,
                        session,
                        gpu.config(),
                        gpu.weights(),
                        &ids,
                        &mask,
                        DeltaKernel::Cuda,
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
        let samples: Vec<f64> = (0..iters)
            .map(|_| {
                let timer = Instant::now();
                std::hint::black_box(run());
                timer.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        let value = stats(samples);
        eprintln!(
            "parcel_full_L256: cuda median {:.2} ms (first {:.0} ms)",
            value["ms_median"].as_f64().unwrap(),
            first_ms
        );
        cases.push(
            json!({"name": "parcel_full_L256", "cuda": value, "cuda_first_call_ms": first_ms}),
        );
    }
    let out = json!({
        "runtime": "rust-cuda",
        "checkpoint": dir,
        "load_ms": load_ms,
        "warmup": warmup,
        "rayon_threads": rayon::current_num_threads(),
        "cases": cases,
    });
    println!("{}", serde_json::to_string(&out).unwrap());
}
