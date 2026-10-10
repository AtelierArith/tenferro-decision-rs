#![cfg(feature = "cuda")]

//! CUDA hardware gates for the Jeff tenferro forward (`--features cuda`).
//!
//! Ignored by default: they need a CUDA device, NVRTC, cuBLAS and cuTENSOR.
//! A missing device is a failure, never a skipped parity result. Run with
//!
//! ```text
//! CUDA_VISIBLE_DEVICES=1 cargo test --release -p jeff-infer --features cuda \
//!     --test cuda -- --ignored --test-threads=1
//! ```
//!
//! The production gate additionally needs the pinned Jeff snapshot in the Hub
//! cache (`hf-fetch jeff`) and skips when it is absent.

use std::path::PathBuf;

use decision_core::PreparedState;
use hf_fetch::{CheckpointSpec, Hub};
use jeff_infer::engine::{Device, JeffBackend, JeffEngine};
use jeff_infer::model::{
    AttentionWeights, CudaDeltaWorkspaces, DeltaKernel, FullAttentionWeights, JeffConfig,
    JeffWeights, LayerWeights, MlpWeights, forward_reference, forward_tenferro_device,
};
use tenferro_gated_delta::{Algorithm, GatedDeltaConfig, GatedDeltaWeights, GatedDeltaWorkspace};
use tenferro_infer::TensorCache;

struct Lcg(u64);

impl Lcg {
    fn fill(&mut self, len: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..len)
            .map(|_| {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                lo + (hi - lo) * (((self.0 >> 40) as f32) / (1u64 << 24) as f32)
            })
            .collect()
    }
}

/// A small mixed stack (DeltaNet, full attention, DeltaNet) with grouped
/// DeltaNet heads and partial RoPE.
fn synthetic(seed: u64) -> (JeffConfig, JeffWeights) {
    let cfg = JeffConfig {
        hidden: 16,
        heads: 2,
        head_dim: 8,
        intermediate: 24,
        eps: 1e-6,
    };
    let mut rng = Lcg(seed);
    let width = cfg.heads * cfg.head_dim;
    let delta = |rng: &mut Lcg| {
        let config = GatedDeltaConfig {
            hidden: cfg.hidden,
            key_heads: 1,
            value_heads: 2,
            key_dim: 8,
            value_dim: 4,
            conv_taps: 4,
            eps: cfg.eps,
            chunk_size: 64,
            algorithm: Algorithm::Chunked,
        };
        let channels =
            2 * config.key_dim * config.key_heads + config.value_dim * config.value_heads;
        let value_width = config.value_dim * config.value_heads;
        AttentionWeights::Delta {
            weights: GatedDeltaWeights {
                qkv: rng.fill(cfg.hidden * channels, -0.3, 0.3),
                z: rng.fill(cfg.hidden * value_width, -0.3, 0.3),
                a: rng.fill(cfg.hidden * config.value_heads, -0.3, 0.3),
                b: rng.fill(cfg.hidden * config.value_heads, -0.3, 0.3),
                conv: rng.fill(config.conv_taps * channels, -0.5, 0.5),
                a_decay: rng.fill(config.value_heads, -1.0, -0.05),
                dt_bias: rng.fill(config.value_heads, -0.5, 0.5),
                norm: rng.fill(config.value_dim, 0.5, 1.5),
                out_proj: rng.fill(value_width * cfg.hidden, -0.3, 0.3),
            },
            config,
        }
    };
    let full = |rng: &mut Lcg| {
        AttentionWeights::Full(FullAttentionWeights {
            q: rng.fill(cfg.hidden * width, -0.3, 0.3),
            gate: rng.fill(cfg.hidden * width, -0.3, 0.3),
            k: rng.fill(cfg.hidden * width, -0.3, 0.3),
            v: rng.fill(cfg.hidden * width, -0.3, 0.3),
            o: rng.fill(width * cfg.hidden, -0.3, 0.3),
            q_norm: rng.fill(cfg.head_dim, -0.5, 0.5),
            k_norm: rng.fill(cfg.head_dim, -0.5, 0.5),
            rope_theta: 10_000_000.0,
            rotary_dim: 4,
        })
    };
    let layer = |rng: &mut Lcg, attention| LayerWeights {
        input_norm: rng.fill(cfg.hidden, -0.5, 0.5),
        post_norm: rng.fill(cfg.hidden, -0.5, 0.5),
        attention,
        mlp: MlpWeights {
            gate: rng.fill(cfg.hidden * cfg.intermediate, -0.3, 0.3),
            up: rng.fill(cfg.hidden * cfg.intermediate, -0.3, 0.3),
            down: rng.fill(cfg.intermediate * cfg.hidden, -0.3, 0.3),
        },
    };
    let first = delta(&mut rng);
    let first = layer(&mut rng, first);
    let second = full(&mut rng);
    let second = layer(&mut rng, second);
    let third = delta(&mut rng);
    let third = layer(&mut rng, third);
    let vocab = 32;
    let options = 5;
    let weights = JeffWeights {
        embedding: rng.fill(cfg.hidden * vocab, -1.0, 1.0),
        vocab,
        layers: vec![first, second, third],
        final_norm: rng.fill(cfg.hidden, -0.5, 0.5),
        readout: rng.fill(cfg.hidden * options, -0.3, 0.3),
        options,
    };
    (cfg, weights)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            assert!(x.is_finite() && y.is_finite());
            (x - y).abs()
        })
        .fold(0.0, f32::max)
}

#[test]
#[ignore = "requires CUDA hardware, NVRTC, cuBLAS and cuTENSOR; run explicitly with --ignored"]
fn cuda_forward_matches_reference_and_reuses_device_state() {
    let (cfg, weights) = synthetic(7);
    let runtime = Device::Cuda(0).runtime().expect("CUDA device required");
    let mut workspace = GatedDeltaWorkspace::new();
    let mut cuda = CudaDeltaWorkspaces::new();
    let mut cache = TensorCache::new();
    let mut cached_entries = None;
    // Lengths across the 64-token chunk boundary, a single token, mask holes
    // and a shrinking length after a longer request (workspace resize).
    for (length, holes) in [
        (7usize, false),
        (7, true),
        (65, true),
        (1, false),
        (130, false),
        (33, true),
    ] {
        let ids: Vec<i64> = (0..length).map(|t| ((t * 7 + 3) % 32) as i64).collect();
        let mask: Vec<f32> = (0..length)
            .map(|t| if holes && t % 5 == 2 { 0.0 } else { 1.0 })
            .collect();
        let expected = forward_reference(&cfg, &weights, &ids, &mask).unwrap();
        for kernel in [DeltaKernel::Cuda, DeltaKernel::TensorNative] {
            let got = runtime
                .with_eager_session(|session| {
                    forward_tenferro_device(
                        &mut workspace,
                        &mut cuda,
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
                .unwrap();
            let diff = max_diff(&got, &expected);
            eprintln!("length {length} holes {holes} {kernel:?}: max diff {diff:e}");
            assert!(diff <= 1e-4, "length {length} {kernel:?}: diff {diff}");
        }
        // Two Delta layers retain one workspace each.
        assert_eq!(cuda.len(), 2);
        // Weights are prepared once; later requests only add nothing new
        // beyond the first request's tensors and scalars.
        match cached_entries {
            None => cached_entries = Some(cache.len()),
            Some(entries) => assert!(cache.len() <= entries + 8, "cache keeps growing"),
        }
    }
}

#[test]
#[ignore = "requires CUDA hardware, NVRTC, cuBLAS and cuTENSOR; run explicitly with --ignored"]
fn cuda_engine_matches_cpu_engine_and_rejects_host_kernel() {
    let (cfg, weights) = synthetic(19);
    let decision = jeff_infer::config::DecisionConfig {
        format_version: 1,
        temperature: 1.0,
        max_options: 5,
    };
    let mut cpu =
        JeffEngine::with_backend(cfg, decision.clone(), weights.clone(), JeffBackend::HostOpt)
            .unwrap();
    let mut gpu = JeffEngine::with_backend(cfg, decision, weights, JeffBackend::Host)
        .unwrap()
        .with_device(Device::Cuda(0))
        .expect("CUDA device required");
    assert_eq!(gpu.backend(), JeffBackend::Tenferro);
    assert_eq!(gpu.delta_kernel(), DeltaKernel::Cuda);
    assert_eq!(gpu.device(), Device::Cuda(0));
    // Left padding is trimmed by the engine; rows differ in length.
    let state = PreparedState {
        input_ids: vec![
            vec![0, 0, 4, 9, 11, 2, 30, 5],
            vec![3, 1, 4, 1, 5, 9, 2, 6],
            vec![0, 0, 0, 0, 0, 0, 0, 17],
        ],
        attention_mask: vec![
            vec![false, false, true, true, true, false, true, true],
            vec![true; 8],
            vec![false, false, false, false, false, false, false, true],
        ],
    };
    for _ in 0..2 {
        let expected = cpu.logits(&state).unwrap();
        let got = gpu.logits(&state).unwrap();
        for (row, (a, b)) in got.iter().zip(&expected).enumerate() {
            let diff = max_diff(a, b);
            eprintln!("row {row}: max diff {diff:e}");
            assert!(diff <= 1e-4, "row {row}: diff {diff}");
        }
    }
    // The CPU-only host recurrent extension reports an error on CUDA.
    let mut host_kernel = gpu.with_delta_kernel(DeltaKernel::HostRecurrent);
    assert!(host_kernel.logits(&state).is_err());
}

/// The `mstrasser/Jeff-Qwen3.5-0.8B` commit pinned by the references.
const JEFF_REVISION: &str = "0f212b3e72acb4dde3f7da61e925d6ab7f819990";

fn checkpoint_dir() -> Option<PathBuf> {
    let mut spec = CheckpointSpec::jeff();
    spec.revision = JEFF_REVISION.to_string();
    let mut hub = Hub::from_env();
    hub.offline = true;
    hub.resolve(&spec).ok()
}

fn json(path: &str) -> serde_json::Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path);
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn f32s(value: &serde_json::Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

#[test]
#[ignore = "loads the ~1.7 GB production Jeff checkpoint on CUDA; run with --release -- --ignored"]
fn cuda_production_logits_match_cpu_and_references() {
    let Some(dir) = checkpoint_dir() else {
        eprintln!("skipping: production Jeff checkpoint is not present");
        return;
    };
    let mut cpu = JeffEngine::load(&dir).unwrap();
    let mut gpu = JeffEngine::load(&dir)
        .unwrap()
        .with_device(Device::Cuda(0))
        .expect("CUDA device required");

    // Julia reference (L8) and the JeffClient.jl parcel case (B1/L256, 101
    // active tokens after left padding; independent PyTorch logits).
    let julia = json("../../fixtures/jeff-real/reference.json");
    let parcel = json("../../extern/JeffClient.jl/examples/data/parcel_reference.json");
    let parcel = &parcel[0];
    let cases = [
        (
            "julia-L8",
            PreparedState {
                input_ids: vec![
                    julia["input_ids"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_i64().unwrap())
                        .collect(),
                ],
                attention_mask: vec![
                    julia["attention_mask"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap() != 0.0)
                        .collect(),
                ],
            },
            f32s(&julia["logits"]),
        ),
        (
            "parcel-L256",
            PreparedState {
                input_ids: vec![
                    parcel["inputs"]["input_ids"][0]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_i64().unwrap())
                        .collect(),
                ],
                attention_mask: vec![
                    parcel["inputs"]["attention_mask"][0]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_i64().unwrap() != 0)
                        .collect(),
                ],
            },
            f32s(&parcel["logits"][0]),
        ),
    ];
    for (name, state, reference) in cases {
        let host = cpu.logits(&state).unwrap().remove(0);
        // Repeat to exercise retained device weights and layer workspaces.
        for request in 0..2 {
            let device = gpu.logits(&state).unwrap().remove(0);
            let scale = reference.iter().map(|v| v.abs()).fold(1.0f32, f32::max);
            let vs_cpu = max_diff(&device, &host);
            let vs_reference = max_diff(&device, &reference);
            let cpu_vs_reference = max_diff(&host, &reference);
            eprintln!(
                "{name} request {request}: cuda-vs-cpu {vs_cpu:e}, cuda-vs-reference \
                 {vs_reference:e}, cpu-vs-reference {cpu_vs_reference:e} (scale {scale})"
            );
            assert!(vs_cpu <= 1e-4 * scale, "{name}: CUDA vs CPU diff {vs_cpu}");
            assert!(
                vs_reference <= cpu_vs_reference + 1e-4 * scale,
                "{name}: CUDA vs reference diff {vs_reference}"
            );
        }
    }
}
