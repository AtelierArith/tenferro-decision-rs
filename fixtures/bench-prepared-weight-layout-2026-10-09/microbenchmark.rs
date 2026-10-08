//! Diagnostic only: compare natural and prepared weight layouts in SGEMM.
use rayon::prelude::*;
use std::time::Instant;

fn prepared_gemm(x: &[f32], rows: usize, k: usize, w: &[f32], n: usize, y: &mut [f32]) {
    y.fill(0.0);
    let block = n.div_ceil(rayon::current_num_threads() * 4).max(8).min(n);
    let xp = x.as_ptr() as usize;
    let wp = w.as_ptr() as usize;
    let yp = y.as_mut_ptr() as usize;
    (0..n.div_ceil(block)).into_par_iter().for_each(|index| {
        let start = index * block;
        // Each task owns disjoint output columns; inputs remain alive/read-only.
        unsafe {
            matrixmultiply::sgemm(
                rows,
                k,
                (n - start).min(block),
                1.0,
                xp as *const f32,
                k as isize,
                1,
                (wp as *const f32).add(start),
                n as isize,
                1,
                1.0,
                (yp as *mut f32).add(start),
                n as isize,
                1,
            );
        }
    });
}

fn values(len: usize) -> Vec<f32> {
    let mut state = 17u64;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 40) as f32 / 16777216.0 - 0.5) * 0.05
        })
        .collect()
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() {
    println!(
        "{{\"rayon_threads\":{},\"warmup\":5,\"iterations\":25,\"rows\":[",
        rayon::current_num_threads()
    );
    let mut first = true;
    for (k, n) in [
        (1024usize, 3072usize),
        (1024, 1024),
        (1024, 5248),
        (2624, 1024),
    ] {
        let natural = values(k * n);
        let begin = Instant::now();
        let mut packed = vec![0.0f32; k * n];
        for i in 0..k {
            for j in 0..n {
                packed[i * n + j] = natural[j * k + i];
            }
        }
        let preparation_ms = begin.elapsed().as_secs_f64() * 1000.0;
        for rows in [8usize, 16, 64] {
            let x = values(rows * k);
            let mut old = vec![0.0f32; rows * n];
            let mut new = old.clone();
            let mut a = Vec::new();
            let mut b = Vec::new();
            for sample in 0..30 {
                // Alternate order to reduce ordering/thermal bias.
                for prepared in if sample % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let begin = Instant::now();
                    if prepared {
                        prepared_gemm(&x, rows, k, &packed, n, &mut new);
                    } else {
                        cpu_kernels::input_mul_weight_transpose_into(
                            &x, rows, k, &natural, n, &mut old,
                        );
                    }
                    let elapsed = begin.elapsed().as_secs_f64() * 1000.0;
                    if sample >= 5 {
                        if prepared {
                            b.push(elapsed);
                        } else {
                            a.push(elapsed);
                        }
                    }
                }
            }
            let max_error = old
                .iter()
                .zip(&new)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(old.iter().chain(&new).all(|v| v.is_finite()));
            assert!(max_error <= 1e-5);
            if !first {
                println!(",");
            }
            first = false;
            println!(
                "{{\"m\":{rows},\"k\":{k},\"n\":{n},\"natural_ms\":{},\"prepared_ms\":{},\"preparation_ms\":{preparation_ms},\"max_abs_error\":{max_error},\"natural_samples_ms\":{a:?},\"prepared_samples_ms\":{b:?}}}",
                median(&mut a.clone()),
                median(&mut b.clone())
            );
        }
    }
    println!("]}}");
}
