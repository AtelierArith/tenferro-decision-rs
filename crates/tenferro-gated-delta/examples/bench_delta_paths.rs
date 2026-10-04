//! Compare the two Gated DeltaNet executions we have:
//!
//! - `delta_layer_tenferro_native`: the head-batched, tensor-native chunked
//!   formulation (the current tenferro-first path; many eager ops + solves).
//! - `delta_layer_recurrent`: the fused host recurrent kernel (used by the host
//!   reference path).
//!
//!     RAYON_NUM_THREADS=8 cargo run --release -p tenferro-gated-delta --example bench_delta_paths

use std::time::Instant;

use tenferro_ad::{EagerRuntime, Tensor as AdTensor};
use tenferro_cpu::CpuBackend;
use tenferro_gated_delta::config::{Algorithm, GatedDeltaConfig};
use tenferro_gated_delta::layer::GatedDeltaWeights;
use tenferro_gated_delta::recurrent::delta_layer_recurrent;
use tenferro_gated_delta::tensor_layer::{
    delta_layer_tenferro_native, prepare_kernel_weights, prepare_tensor_weights,
};
use tenferro_gated_delta::workspace::GatedDeltaWorkspace;
use tenferro_gated_delta::{EagerSessionGatedDeltaExt, GatedDeltaOp};
use tenferro_infer::TensorCache;

fn col_major(rows: usize, cols: usize, row_major: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            out[row + col * rows] = row_major[row * cols + col];
        }
    }
    out
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32 - 0.5
    }
    fn fill(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

fn main() {
    let cfg = GatedDeltaConfig {
        hidden: 1024,
        key_dim: 128,
        value_dim: 128,
        key_heads: 8,
        value_heads: 16,
        conv_taps: 4,
        eps: 1e-5,
        chunk_size: 64,
        algorithm: Algorithm::Recurrent,
    };
    let key_width = cfg.key_dim * cfg.key_heads;
    let value_width = cfg.value_dim * cfg.value_heads;
    let conv_channels = 2 * key_width + value_width;
    let mut rng = Lcg(7);
    let weights = GatedDeltaWeights {
        qkv: rng.fill(cfg.hidden * conv_channels),
        z: rng.fill(cfg.hidden * value_width),
        a: rng.fill(cfg.hidden * cfg.value_heads),
        b: rng.fill(cfg.hidden * cfg.value_heads),
        conv: rng.fill(cfg.conv_taps * conv_channels),
        a_decay: (0..cfg.value_heads)
            .map(|_| -0.1 - rng.next().abs())
            .collect(),
        dt_bias: rng.fill(cfg.value_heads),
        norm: (0..cfg.value_dim).map(|_| 0.5 + rng.next().abs()).collect(),
        out_proj: rng.fill(value_width * cfg.hidden),
    };

    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let iters = 10;

    println!("Gated DeltaNet layer: tensor-native (chunked) vs host recurrent");
    for length in [8usize, 16, 64] {
        let mask = vec![1.0f32; length];
        let x = rng.fill(cfg.hidden * length);

        // host recurrent
        let mut ws = GatedDeltaWorkspace::new();
        let _ = delta_layer_recurrent(&cfg, &weights, &x, &mask, &mut ws).unwrap();
        let t = Instant::now();
        for _ in 0..iters {
            let _ = delta_layer_recurrent(&cfg, &weights, &x, &mask, &mut ws).unwrap();
        }
        let host_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

        // tensor-native chunked
        let mut cache = TensorCache::new();
        let mut run = || {
            runtime
                .with_eager_session(|session| {
                    let tw = prepare_tensor_weights(session, &cfg, &weights, &mut cache)?;
                    let x_t = session.constant_from(AdTensor::from_vec_col_major(
                        vec![cfg.hidden, length],
                        col_major(cfg.hidden, length, &x),
                    )?)?;
                    delta_layer_tenferro_native(session, &cfg, &tw, &x_t, &mask)
                })
                .unwrap()
                .unwrap()
        };
        let _ = run();
        let t = Instant::now();
        for _ in 0..iters {
            let _ = run();
        }
        let ten_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

        // host recurrent via the `GatedDelta` extension op (session op; weights
        // are fed in the kernel's row-major layout, so nothing is transposed)
        let mut run_ext = || {
            runtime
                .with_eager_session(|session| {
                    let kw = prepare_kernel_weights(session, &cfg, &weights, &mut cache)?;
                    let x_t = session.constant_from(AdTensor::from_vec_col_major(
                        vec![length, cfg.hidden],
                        x.clone(),
                    )?)?;
                    let mask_t = session
                        .constant_from(AdTensor::from_vec_col_major(vec![length], mask.clone())?)?;
                    let op = GatedDeltaOp::from_config(&cfg);
                    session.gated_delta(
                        op,
                        &[
                            &x_t,
                            &mask_t,
                            &kw.qkv,
                            &kw.z,
                            &kw.a,
                            &kw.b,
                            &kw.conv,
                            &kw.a_decay,
                            &kw.dt_bias,
                            &kw.norm,
                            &kw.out_proj,
                        ],
                    )
                })
                .unwrap()
                .unwrap()
        };
        let _ = run_ext();
        let t = Instant::now();
        for _ in 0..iters {
            let _ = run_ext();
        }
        let ext_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

        println!(
            "len={length:3}: host recurrent {host_ms:8.3} ms | tensor-native {ten_ms:8.3} ms | ext-op {ext_ms:8.3} ms | tensor/host {:.2}x | ext/host {:.2}x",
            ten_ms / host_ms,
            ext_ms / host_ms
        );
    }
}
