//! Compare the host GEMM providers on the projection shapes the forwards use:
//!
//! - `cpu_kernels::matmul_row_major_into` (Accelerate on macOS above the
//!   threshold, `matrixmultiply` below)
//! - direct `faer` (what tenferro-cpu's default `dot_general` uses)
//!
//!     RAYON_NUM_THREADS=8 cargo run --release -p bench-suite --example host_gemm_providers

use std::time::Instant;

use faer::{Accum, MatMut, MatRef, Par};

fn main() {
    let iters = 50usize;
    // (in, out, length): DeltaNet qkv/z/out projections and MLP projections.
    let shapes = [
        (1024usize, 4096usize, 8usize),
        (1024, 2048, 8),
        (2048, 1024, 8),
        (1024, 1024, 8),
        (4096, 1024, 8),
        (1024, 4096, 64),
        (4096, 1024, 512),
    ];
    println!("host GEMM providers: cpu-kernels (Accelerate) vs faer");
    for (in_dim, out_dim, length) in shapes {
        let w: Vec<f32> = (0..in_dim * out_dim)
            .map(|i| (i % 97) as f32 * 0.01)
            .collect();
        let x: Vec<f32> = (0..in_dim * length)
            .map(|i| (i % 89) as f32 * 0.01)
            .collect();

        // cpu-kernels
        let mut y_host = vec![0.0f32; out_dim * length];
        let run_host = |y: &mut [f32]| {
            cpu_kernels::matmul_row_major_into(&w, in_dim, out_dim, &x, length, y);
        };
        run_host(&mut y_host);
        let t = Instant::now();
        for _ in 0..iters {
            run_host(&mut y_host);
        }
        let host_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

        // faer: y (out,length) = Wᵀ (out,in) · x (in,length)
        let mut y_faer = vec![0.0f32; out_dim * length];
        let run_faer = |y: &mut [f32]| {
            // SAFETY: the slices are exactly the shape arguments and do not
            // overlap.
            unsafe {
                let w_ref =
                    MatRef::from_raw_parts(w.as_ptr(), in_dim, out_dim, out_dim as isize, 1);
                let w_t = w_ref.transpose(); // (out, in)
                let x_ref = MatRef::from_raw_parts(x.as_ptr(), in_dim, length, length as isize, 1);
                let mut c =
                    MatMut::from_raw_parts_mut(y.as_mut_ptr(), out_dim, length, length as isize, 1);
                faer::linalg::matmul::matmul(
                    &mut c,
                    Accum::Replace,
                    &w_t,
                    &x_ref,
                    1.0f32,
                    Par::rayon(0),
                );
            }
        };
        run_faer(&mut y_faer);
        let t = Instant::now();
        for _ in 0..iters {
            run_faer(&mut y_faer);
        }
        let faer_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

        let diff = y_host
            .iter()
            .zip(&y_faer)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "in={in_dim:5} out={out_dim:5} len={length:3}: host {host_ms:8.3} ms | faer {faer_ms:8.3} ms | faer/host {:.2}x | maxdiff {diff:.2e}",
            faer_ms / host_ms
        );
    }
}
