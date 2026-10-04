//! Minimal working example: tenferro's fused elementwise region is ~3x slower
//! than eager per-op kernels for a long elementwise chain on CPU.
//!
//! Run:
//!     cargo run --release -p bench-suite --example fused_elementwise_mwe -- <K> <ITERS>
//! e.g.
//!     RAYON_NUM_THREADS=8 cargo run --release -p bench-suite --example fused_elementwise_mwe -- 200 20
//!
//! The traced/compiled program is one elementwise region covering all `K`
//! `tanh` ops. `PreparedCompiledGraph::elementwise_region_summary` /
//! `_execution_counts` show the region is executed as a single fused command
//! (`fused == runs`, `fallback == 0`), yet the total time scales linearly with
//! `K` and is much slower than eager applying the same ops one by one.

use std::time::Instant;

use tenferro_ad::{EagerRuntime, Tensor};
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{DType, GraphCompiler, Runtime, TracedTensor};

fn main() {
    let mut args = std::env::args().skip(1);
    let k: usize = args.next().map(|a| a.parse().unwrap()).unwrap_or(200);
    let iters: usize = args.next().map(|a| a.parse().unwrap()).unwrap_or(20);

    let hidden = 1024usize;
    let length = 64usize;
    let elements = hidden * length;
    let data: Vec<f32> = (0..elements).map(|i| ((i as f32) * 0.0001).sin()).collect();

    // ---------- compiled: one traced chain, one fused elementwise region ----------
    let x_in = TracedTensor::input_concrete_shape(DType::F32, &[hidden, length]).unwrap();
    let mut y = x_in.clone();
    for _ in 0..k {
        y = y.tanh().unwrap();
    }
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x_in, DType::F32, &[hidden, length])])
        .unwrap();

    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&backend).unwrap())
        .unwrap();
    let runtime = builder.build().unwrap();

    let x_t = Tensor::from_vec_col_major(vec![hidden, length], data.clone()).unwrap();
    let prepared = runtime.prepare_compiled(&program, &[&x_t]).unwrap();
    let _ = runtime.run_prepared(&prepared, &[&x_t]).unwrap(); // warmup
    let start = Instant::now();
    for _ in 0..iters {
        let _ = runtime.run_prepared(&prepared, &[&x_t]).unwrap();
    }
    let compiled_ms = start.elapsed().as_secs_f64() * 1e3 / iters as f64;
    let (regions, covered) = prepared.elementwise_region_summary();
    let (fused, fallback) = prepared.elementwise_region_execution_counts();

    // ---------- eager: the same chain, one kernel per op ----------
    let eager_runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let x_const = eager_runtime
        .with_eager_session(|s| {
            s.constant_from(Tensor::from_vec_col_major(
                vec![hidden, length],
                data.clone(),
            )?)
        })
        .unwrap()
        .unwrap();
    let eager_run = || {
        eager_runtime
            .with_eager_session(|s| {
                let mut y = x_const.clone();
                for _ in 0..k {
                    y = s.tanh(&y)?;
                }
                let _ = s.duplicate_value(&y)?;
                Ok::<(), tenferro_ad::Error>(())
            })
            .unwrap()
            .unwrap();
    };
    eager_run();
    let start = Instant::now();
    for _ in 0..iters {
        eager_run();
    }
    let eager_ms = start.elapsed().as_secs_f64() * 1e3 / iters as f64;

    println!("tenferro fused-elementwise MWE");
    println!("chain = {k} x tanh, tensor = {hidden}x{length} f32, iters = {iters}");
    println!("eager    (per-op kernels):   {eager_ms:9.3} ms");
    println!("compiled (one fused region): {compiled_ms:9.3} ms");
    println!("compiled / eager = {:.2}x", compiled_ms / eager_ms);
    println!(
        "region plan: regions={regions} covered_insts={covered} fused_runs={fused} fallbacks={fallback}"
    );
}
