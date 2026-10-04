//! Minimal working example: the eager `dot_general` path has avoidsable
//! per-call overhead compared with the *same* faer GEMM run directly.
//!
//! Run:
//!     RAYON_NUM_THREADS=8 cargo run --release -p bench-suite --example eager_dot_general_mwe
//!
//! For each projection shape, (a) times `EagerSession::dot_general(x, weight)`
//! (which allocates its output, re-runs the GEMM analysis, and enters the faer
//! provider per call), and (b) times the identical GEMM with `faer` directly
//! into a preallocated buffer with `Par::rayon(0)`. The gap is eager-path
//! overhead on top of the same kernels — i.e. headroom.

use std::time::Instant;

use faer::{Accum, MatMut, MatRef, Par};
use tenferro_ad::{DotGeneralConfig, EagerRuntime, Tensor};
use tenferro_cpu::CpuBackend;

fn lcg(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 40) as f32) / (1u64 << 24) as f32 - 0.5
}

fn col_major(rows: usize, cols: usize, row_major: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            out[row + col * rows] = row_major[row * cols + col];
        }
    }
    out
}

fn dot_cfg() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: [0].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    }
}

fn main() {
    let iters = 100usize;
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();

    println!("eager dot_general vs direct faer GEMM (Par::rayon(0)), iters={iters}");
    let shapes = [
        (1024usize, 1024usize, 8usize),
        (1024, 1024, 64),
        (1024, 1024, 512),
        (4096, 1024, 512),
        (4096, 4096, 512),
    ];
    for &(in_dim, out_dim, length) in &shapes {
        let mut state = 7u64;
        let w_rm: Vec<f32> = (0..in_dim * out_dim).map(|_| lcg(&mut state)).collect();
        let x_rm: Vec<f32> = (0..in_dim * length).map(|_| lcg(&mut state)).collect();
        let w = col_major(in_dim, out_dim, &w_rm);
        let x = col_major(in_dim, length, &x_rm);

        // (a) eager dot_general
        let (x_t, w_t) = runtime
            .with_eager_session(|session| {
                let x_t = session
                    .constant_from(Tensor::from_vec_col_major(vec![in_dim, length], x.clone())?)?;
                let w_t = session.constant_from(Tensor::from_vec_col_major(
                    vec![in_dim, out_dim],
                    w.clone(),
                )?)?;
                Ok::<_, tenferro_ad::Error>((x_t, w_t))
            })
            .unwrap()
            .unwrap();
        let _ = runtime
            .with_eager_session(|session| session.dot_general(&x_t, &w_t, dot_cfg()))
            .unwrap()
            .unwrap();
        let start = Instant::now();
        for _ in 0..iters {
            let _ = runtime
                .with_eager_session(|session| session.dot_general(&x_t, &w_t, dot_cfg()))
                .unwrap()
                .unwrap();
        }
        let eager_ms = start.elapsed().as_secs_f64() * 1e3 / iters as f64;

        // (b) direct faer into a preallocated buffer
        let a = unsafe {
            MatRef::<f32>::from_raw_parts(x.as_ptr(), in_dim, length, 1, in_dim as isize)
        };
        let b = unsafe {
            MatRef::<f32>::from_raw_parts(w.as_ptr(), in_dim, out_dim, 1, in_dim as isize)
        };
        let a_t = a.transpose();
        let mut y = vec![0.0f32; length * out_dim];
        let run_faer = |y: &mut [f32]| {
            // SAFETY: `y` has `length * out_dim` elements; `a_t`/`b` own valid
            // storage for the duration of this call.
            unsafe {
                let mut c = MatMut::<f32>::from_raw_parts_mut(
                    y.as_mut_ptr(),
                    length,
                    out_dim,
                    1,
                    length as isize,
                );
                faer::linalg::matmul::matmul(
                    &mut c,
                    Accum::Replace,
                    &a_t,
                    &b,
                    1.0f32,
                    Par::rayon(0),
                );
            }
        };
        run_faer(&mut y);
        let start = Instant::now();
        for _ in 0..iters {
            run_faer(&mut y);
        }
        let faer_ms = start.elapsed().as_secs_f64() * 1e3 / iters as f64;

        // (c) the host kernel path (Accelerate on macOS, matrixmultiply elsewhere)
        let mut y_host = vec![0.0f32; out_dim * length];
        let run_host = |y: &mut [f32]| {
            cpu_kernels::matmul_row_major_into(&w_rm, in_dim, out_dim, &x_rm, length, y)
        };
        run_host(&mut y_host);
        let start = Instant::now();
        for _ in 0..iters {
            run_host(&mut y_host);
        }
        let host_ms = start.elapsed().as_secs_f64() * 1e3 / iters as f64;

        println!(
            "in={in_dim:5} out={out_dim:5} len={length:3}: eager {eager_ms:7.3} ms | faer {faer_ms:7.3} ms | host {host_ms:7.3} ms | eager/faer {:.2}x | eager/host {:.2}x",
            eager_ms / faer_ms,
            eager_ms / host_ms
        );
    }
}
