//! Compare the eager and graph-compiled Gated DeltaNet layer.
//!
//!     cargo run --release -p tenferro-gated-delta --example bench_compiled_delta
//!
//! Builds one DeltaNet layer with synthetic weights at the Jeff shape, runs the
//! eager tensor-native layer and the `GraphCompiler`-compiled traced layer, and
//! reports both latencies and their max output difference.

use std::time::Instant;

use tenferro_ad::{EagerRuntime, Tensor as AdTensor};
use tenferro_cpu::CpuBackend;
use tenferro_gated_delta::config::{Algorithm, GatedDeltaConfig};
use tenferro_gated_delta::layer::GatedDeltaWeights;
use tenferro_gated_delta::tensor_layer::{delta_layer_tenferro_native, prepare_tensor_weights};
use tenferro_gated_delta::traced_layer::{
    delta_layer_traced, input_mask, input_x, prepare_traced_weights,
};
use tenferro_infer::TensorCache;
use tenferro_runtime::{DType, GraphCompiler, Runtime};

/// Column-major copy of row-major `(rows, cols)` data.
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

fn run_eager(
    runtime: &std::sync::Arc<EagerRuntime>,
    cfg: &GatedDeltaConfig,
    weights: &GatedDeltaWeights,
    x: &[f32],
    mask: &[f32],
    cache: &mut TensorCache,
) -> Vec<f32> {
    let length = mask.len();
    runtime
        .with_eager_session(|session| {
            let tw = prepare_tensor_weights(session, cfg, weights, cache)?;
            let x = session.constant_from(AdTensor::from_vec_col_major(
                vec![cfg.hidden, length],
                col_major(cfg.hidden, length, x),
            )?)?;
            let out = delta_layer_tenferro_native(session, cfg, &tw, &x, mask)?;
            let host = session.duplicate_value(&out)?;
            Ok::<Vec<f32>, tenferro_ad::Error>(host.as_slice::<f32>()?.to_vec())
        })
        .unwrap()
        .unwrap()
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
        algorithm: Algorithm::Chunked,
    };
    let length = 64usize;
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
    let x = rng.fill(cfg.hidden * length);
    let mask = vec![1.0f32; length];

    let iters = 5;

    // ---- eager tensor-native ----
    let eager_runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let mut cache = TensorCache::new();
    let eager = run_eager(&eager_runtime, &cfg, &weights, &x, &mask, &mut cache);
    let t = Instant::now();
    for _ in 0..iters {
        let _ = run_eager(&eager_runtime, &cfg, &weights, &x, &mask, &mut cache);
    }
    let eager_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

    // ---- graph-compiled ----
    let traced_weights = prepare_traced_weights(&cfg, &weights).unwrap();
    let x_in = input_x(cfg.hidden, length).unwrap();
    let mask_in = input_mask(length).unwrap();
    let out = delta_layer_traced(&cfg, &traced_weights, &x_in, &mask_in, length).unwrap();

    let mut compiler = GraphCompiler::new();
    let t = Instant::now();
    let program = compiler
        .compile_with_input_specs(
            &out,
            &[
                (&x_in, DType::F32, &[cfg.hidden, length]),
                (&mask_in, DType::F32, &[length]),
            ],
        )
        .unwrap();
    let compile_ms = t.elapsed().as_secs_f64() * 1e3;

    let mut builder = Runtime::builder();
    let backend = CpuBackend::new();
    let engine_id = tenferro_cpu::runtime_engine_id().unwrap();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&backend).unwrap())
        .unwrap();
    builder
        .install_extension_module(
            tenferro_linalg::extension_module::<CpuBackend>(engine_id).unwrap(),
        )
        .unwrap();
    let runtime = builder.build().unwrap();

    let x_t =
        AdTensor::from_vec_col_major(vec![cfg.hidden, length], col_major(cfg.hidden, length, &x))
            .unwrap();
    let mask_t = AdTensor::from_vec_col_major(vec![length], mask.clone()).unwrap();

    let _ = runtime.run_compiled(&program, &[&x_t, &mask_t]).unwrap();
    let prepared = runtime
        .prepare_compiled(&program, &[&x_t, &mask_t])
        .unwrap();
    let t = Instant::now();
    let mut compiled_out = Vec::new();
    for _ in 0..iters {
        compiled_out = runtime.run_prepared(&prepared, &[&x_t, &mask_t]).unwrap();
    }
    let compiled_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

    let compiled_host = compiled_out[0].as_slice::<f32>().unwrap();
    let diff = eager
        .iter()
        .zip(compiled_host)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!(
        "shape: hidden={} length={length} heads={}",
        cfg.hidden, cfg.value_heads
    );
    println!("eager native:   {eager_ms:.2} ms/iter");
    println!("compiled:       {compiled_ms:.2} ms/iter  (compile {compile_ms:.1} ms)");
    println!("speedup:        {:.2}x", eager_ms / compiled_ms);
    println!("max |eager - compiled| = {diff:.3e}");

    elementwise_bench(iters);
}

/// An elementwise-only chain: if the compiled path fuses it, total time stays
/// roughly flat as the chain grows; if it runs op-by-op, time scales with `k`.
fn elementwise_bench(iters: usize) {
    let hidden = 1024usize;
    let length = 64usize;
    let mut rng = Lcg(99);
    let x = rng.fill(hidden * length);
    let x_t =
        AdTensor::from_vec_col_major(vec![hidden, length], col_major(hidden, length, &x)).unwrap();
    let eager_runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();

    for k in [1usize, 10, 100, 200] {
        let eager =
            |session: &mut tenferro_ad::EagerSession<'_>| -> Result<Vec<f32>, tenferro_ad::Error> {
                let mut y = session.constant_from(AdTensor::from_vec_col_major(
                    vec![hidden, length],
                    col_major(hidden, length, &x),
                )?)?;
                for _ in 0..k {
                    y = session.tanh(&y)?;
                }
                let host = session.duplicate_value(&y)?;
                Ok(host.as_slice::<f32>()?.to_vec())
            };
        let _ = eager_runtime.with_eager_session(eager).unwrap().unwrap();
        let t = Instant::now();
        for _ in 0..iters {
            let _ = eager_runtime.with_eager_session(eager).unwrap().unwrap();
        }
        let eager_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

        let x_in =
            tenferro_runtime::TracedTensor::input_concrete_shape(DType::F32, &[hidden, length])
                .unwrap();
        let mut y = x_in.clone();
        for _ in 0..k {
            y = y.tanh().unwrap();
        }
        let mut compiler = GraphCompiler::new();
        let program = compiler
            .compile_with_input_specs(&y, &[(&x_in, DType::F32, &[hidden, length])])
            .unwrap();
        let mut builder = Runtime::builder();
        builder
            .register_engine(tenferro_cpu::runtime_engine_registration(&CpuBackend::new()).unwrap())
            .unwrap();
        let runtime = builder.build().unwrap();
        let prepared = runtime.prepare_compiled(&program, &[&x_t]).unwrap();
        let _ = runtime.run_prepared(&prepared, &[&x_t]).unwrap();
        let t = Instant::now();
        for _ in 0..iters {
            let _ = runtime.run_prepared(&prepared, &[&x_t]).unwrap();
        }
        let compiled_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

        let (planned, covered) = prepared.elementwise_region_summary();
        let (fused, fallback) = prepared.elementwise_region_execution_counts();
        println!(
            "elementwise k={k:3} ({hidden}x{length}): eager {eager_ms:8.3} ms  compiled {compiled_ms:8.3} ms  regions={planned} covered_insts={covered} fused={fused} fallback={fallback}"
        );
    }
}
