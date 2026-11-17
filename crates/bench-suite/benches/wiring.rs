//! Phase 0 benchmark harness skeleton.
//!
//! Measures a single CPU `matmul` through the tenferro session API so the
//! harness, metadata capture, and reporting path are exercised end to end. Real
//! primitive and model benchmarks replace this in later phases.

use bench_suite::BenchMetadata;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use tenferro_runtime::{Tensor, TensorSessionOpsExt};

fn matmul_benchmark(c: &mut Criterion) {
    // Attach build/machine provenance to the run; real runners write this next
    // to the criterion output.
    eprintln!(
        "benchmark metadata:\n{}",
        BenchMetadata::capture().to_json()
    );

    let mut group = c.benchmark_group("tenferro_cpu_matmul");
    for &size in &[16usize, 64, 256] {
        let a = Tensor::from_vec_col_major(vec![size, size], vec![1.0_f32; size * size]).unwrap();
        let b = Tensor::from_vec_col_major(vec![size, size], vec![1.0_f32; size * size]).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            let mut engine = tenferro_infer::cpu_engine();
            bencher.iter(|| {
                let out = a.matmul(&b, &mut engine).unwrap();
                std::hint::black_box(out.shape());
            });
        });
    }
    group.finish();
}

criterion_group! {
    name = wiring;
    config = Criterion::default();
    targets = matmul_benchmark
}
criterion_main!(wiring);
