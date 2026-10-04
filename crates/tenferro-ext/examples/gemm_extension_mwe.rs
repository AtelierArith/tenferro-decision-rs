//! Minimal working example: the self-hosted `tenferro-ext` GEMM extension gives
//! the eager session a faster CPU path than eager `dot_general`.
//!
//! Run:
//!     RAYON_NUM_THREADS=8 cargo run --release -p tenferro-ext --example gemm_extension_mwe
//!
//! For each projection shape `(in, out, length)`, (a) times
//! `EagerSession::dot_general(x, weight, contract 0/0)` and (b) times the
//! `EagerSessionGemmExt::gemm` extension op, both on the same `f32` data with a
//! warmup and several iterations. The two compute the same set of products —
//! `dot_general` yields the logical `(length, out)` orientation, the extension
//! yields its transpose `(out, length)` — so the ratio reflects path overhead,
//! not a difference in FLOPs.

use std::time::Instant;

use tenferro_ad::{DotGeneralConfig, EagerRuntime, Tensor};
use tenferro_cpu::CpuBackend;
use tenferro_ext::EagerSessionGemmExt;

fn lcg(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 40) as f32) / (1u64 << 24) as f32 - 0.5
}

/// Reinterpret a row-major `(rows, cols)` buffer as column-major storage.
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
    let iters = 30usize;
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();

    println!(
        "eager dot_general vs tenferro-ext GEMM (cpu-kernels), iters={iters} \
         (dot_general -> (length, out); gemm -> (out, length))"
    );
    println!(
        "{:>6} {:>6} {:>5} | {:>11} {:>11} | {:>7}",
        "in", "out", "len", "dot ms", "gemm ms", "ratio"
    );

    let shapes = [
        (1024usize, 1024usize, 64usize),
        (1024, 4096, 8),
        (2048, 1024, 64),
    ];
    for &(in_dim, out_dim, length) in &shapes {
        let mut state = 7u64;
        let w_rm: Vec<f32> = (0..in_dim * out_dim).map(|_| lcg(&mut state)).collect();
        let x_rm: Vec<f32> = (0..in_dim * length).map(|_| lcg(&mut state)).collect();
        let w = col_major(in_dim, out_dim, &w_rm);
        let x = col_major(in_dim, length, &x_rm);

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

        // Warmup both paths once (amortize planning/caching).
        let _ = runtime
            .with_eager_session(|session| session.dot_general(&x_t, &w_t, dot_cfg()))
            .unwrap()
            .unwrap();
        let _ = runtime
            .with_eager_session(|session| session.gemm(&x_t, &w_t))
            .unwrap()
            .unwrap();

        let start = Instant::now();
        for _ in 0..iters {
            let _ = runtime
                .with_eager_session(|session| session.dot_general(&x_t, &w_t, dot_cfg()))
                .unwrap()
                .unwrap();
        }
        let dot_ms = start.elapsed().as_secs_f64() * 1e3 / iters as f64;

        let start = Instant::now();
        for _ in 0..iters {
            let _ = runtime
                .with_eager_session(|session| session.gemm(&x_t, &w_t))
                .unwrap()
                .unwrap();
        }
        let gemm_ms = start.elapsed().as_secs_f64() * 1e3 / iters as f64;

        println!(
            "{in_dim:>6} {out_dim:>6} {length:>5} | {dot_ms:>11.3} {gemm_ms:>11.3} | {:>6.2}x",
            dot_ms / gemm_ms
        );
    }
}
