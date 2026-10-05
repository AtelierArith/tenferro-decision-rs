//! Host CPU kernels shared by the engines.
//!
//! These run on plain `&[f32]` host slices and have no tenferro dependency, so
//! both the host forwards and the tenferro-backed paths can share one
//! implementation. The matmul kernels are BLAS-class (`matrixmultiply`) and
//! parallelized across output rows with `rayon` when the work is large enough.
//!
//! Layouts follow the engines' convention: weights are row-major `(in, out)`
//! and activations are row-major `(dim, length)`.

use rayon::prelude::*;

/// Below this many multiply-adds the threading overhead is not worth it.
const PARALLEL_THRESHOLD: usize = 1 << 18;

/// Rows per parallel task: enough tasks to fill the pool without tiny chunks.
fn parallel_row_block(rows: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    rows.div_ceil(threads * 4).max(8)
}

/// `y += weightᵀ · x`, with row-major `(in, out)` `weight`, row-major
/// `(in, length)` `x`, and row-major `(out, length)` `y`.
///
/// Shapes must match: `weight.len() == in_dim * out_dim`,
/// `x.len() == in_dim * length`, `y.len() == out_dim * length`.
pub fn matmul_row_major_add_into(
    weight: &[f32],
    in_dim: usize,
    out_dim: usize,
    x: &[f32],
    length: usize,
    y: &mut [f32],
) {
    debug_assert_eq!(weight.len(), in_dim * out_dim);
    debug_assert_eq!(x.len(), in_dim * length);
    debug_assert_eq!(y.len(), out_dim * length);
    if out_dim == 0 || length == 0 || in_dim == 0 {
        return;
    }
    // y ← weightᵀ x + y. `weightᵀ` is `(out, in)` with row stride 1 (the `out`
    // axis of the row-major `(in, out)` storage) and column stride `out_dim`.
    let work = in_dim.saturating_mul(out_dim).saturating_mul(length);
    // This kernel blocks over `out_dim`, which is large for every Jeff /
    // DeltaNet / MLP projection, so the rayon row-block path below beats
    // Accelerate's per-call dispatch overhead. Do not route it through BLAS.
    if work >= PARALLEL_THRESHOLD && out_dim > 1 {
        let block = parallel_row_block(out_dim);
        y.par_chunks_mut(block * length)
            .enumerate()
            .for_each(|(index, y_block)| {
                let rows = y_block.len() / length;
                // SAFETY: each task owns a disjoint, contiguous block of `y`
                // rows; `weight`/`x` are only read and the block index is in
                // range.
                unsafe {
                    matrixmultiply::sgemm(
                        rows,
                        in_dim,
                        length,
                        1.0,
                        weight.as_ptr().add(index * block),
                        1,
                        out_dim as isize,
                        x.as_ptr(),
                        length as isize,
                        1,
                        1.0,
                        y_block.as_mut_ptr(),
                        length as isize,
                        1,
                    );
                }
            });
    } else {
        unsafe {
            matrixmultiply::sgemm(
                out_dim,
                in_dim,
                length,
                1.0,
                weight.as_ptr(),
                1,
                out_dim as isize,
                x.as_ptr(),
                length as isize,
                1,
                1.0,
                y.as_mut_ptr(),
                length as isize,
                1,
            );
        }
    }
}

/// `y = weightᵀ · x`; see [`matmul_row_major_add_into`] for the layouts.
pub fn matmul_row_major_into(
    weight: &[f32],
    in_dim: usize,
    out_dim: usize,
    x: &[f32],
    length: usize,
    y: &mut [f32],
) {
    debug_assert_eq!(y.len(), out_dim * length);
    y.iter_mut().for_each(|value| *value = 0.0);
    matmul_row_major_add_into(weight, in_dim, out_dim, x, length, y);
}

/// Allocating `y = weightᵀ · x`; see [`matmul_row_major_add_into`].
pub fn matmul_row_major(
    weight: &[f32],
    in_dim: usize,
    out_dim: usize,
    x: &[f32],
    length: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim * length];
    matmul_row_major_into(weight, in_dim, out_dim, x, length, &mut y);
    y
}

/// `y += x · weightᵀ`, with row-major `(rows, in_dim)` `x`, row-major
/// `(out_dim, in_dim)` `weight`, and row-major `(rows, out_dim)` `y`.
///
/// This is the Laya `LinearWeights` layout (`y[o] = Σ_i weight[o, i] · x[i]`).
pub fn input_mul_weight_transpose_add_into(
    x: &[f32],
    rows: usize,
    in_dim: usize,
    weight: &[f32],
    out_dim: usize,
    y: &mut [f32],
) {
    debug_assert_eq!(x.len(), rows * in_dim);
    debug_assert_eq!(weight.len(), out_dim * in_dim);
    debug_assert_eq!(y.len(), rows * out_dim);
    if rows == 0 || out_dim == 0 || in_dim == 0 {
        return;
    }
    // y ← x · weightᵀ + y. `weightᵀ` is `(in_dim, out_dim)` with row stride 1
    // (the `in_dim` axis of the row-major `(out_dim, in_dim)` storage).
    let work = rows.saturating_mul(in_dim).saturating_mul(out_dim);
    if work >= PARALLEL_THRESHOLD && out_dim > 1 {
        // Block over the large `out_dim` axis (columns of `y`) so the rayon
        // tasks parallelize even when `rows` (= batch×sequence, or sequence
        // length) is tiny at decode. Each task writes a disjoint column block.
        let threads = rayon::current_num_threads().max(1);
        let block = out_dim.div_ceil(threads * 4).max(8).min(out_dim);
        let nblocks = out_dim.div_ceil(block);
        let x_addr = x.as_ptr() as usize;
        let w_addr = weight.as_ptr() as usize;
        let y_addr = y.as_mut_ptr() as usize;
        (0..nblocks).into_par_iter().for_each(|bi| {
            let o0 = bi * block;
            let n = (out_dim - o0).min(block);
            // SAFETY: each task writes the disjoint column range `[o0, o0+n)` of
            // every `y` row and reads only `x`/`weight`; the raw addresses just
            // recapture the slices that `rayon` cannot borrow for a strided
            // output.
            unsafe {
                let x_ptr = x_addr as *const f32;
                let w_ptr = (w_addr as *const f32).add(o0 * in_dim);
                let y_ptr = (y_addr as *mut f32).add(o0);
                matrixmultiply::sgemm(
                    rows,
                    in_dim,
                    n,
                    1.0,
                    x_ptr,
                    in_dim as isize,
                    1,
                    w_ptr,
                    1,
                    in_dim as isize,
                    1.0,
                    y_ptr,
                    out_dim as isize,
                    1,
                );
            }
        });
    } else {
        unsafe {
            matrixmultiply::sgemm(
                rows,
                in_dim,
                out_dim,
                1.0,
                x.as_ptr(),
                in_dim as isize,
                1,
                weight.as_ptr(),
                1,
                in_dim as isize,
                1.0,
                y.as_mut_ptr(),
                out_dim as isize,
                1,
            );
        }
    }
}

/// `y = x · weightᵀ`; see [`input_mul_weight_transpose_add_into`].
pub fn input_mul_weight_transpose_into(
    x: &[f32],
    rows: usize,
    in_dim: usize,
    weight: &[f32],
    out_dim: usize,
    y: &mut [f32],
) {
    debug_assert_eq!(y.len(), rows * out_dim);
    y.iter_mut().for_each(|value| *value = 0.0);
    input_mul_weight_transpose_add_into(x, rows, in_dim, weight, out_dim, y);
}

/// `y = x · weightᵀ + bias`, with the `(out_dim,)` `bias` broadcast over the
/// `rows` axis. Row-major `(rows, in_dim)` `x`, row-major `(out_dim, in_dim)`
/// `weight`, row-major `(rows, out_dim)` `y`; see
/// [`input_mul_weight_transpose_add_into`].
pub fn input_mul_weight_transpose_bias_into(
    x: &[f32],
    rows: usize,
    in_dim: usize,
    weight: &[f32],
    out_dim: usize,
    bias: &[f32],
    y: &mut [f32],
) {
    debug_assert_eq!(bias.len(), out_dim);
    for row in 0..rows {
        y[row * out_dim..(row + 1) * out_dim].copy_from_slice(bias);
    }
    input_mul_weight_transpose_add_into(x, rows, in_dim, weight, out_dim, y);
}

/// Allocating `y = x · weightᵀ`; see [`input_mul_weight_transpose_add_into`].
pub fn input_mul_weight_transpose(
    x: &[f32],
    rows: usize,
    in_dim: usize,
    weight: &[f32],
    out_dim: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * out_dim];
    input_mul_weight_transpose_into(x, rows, in_dim, weight, out_dim, &mut y);
    y
}

/// Feature-first LayerNorm: normalize each column of the column-major
/// `(d, cols)` activation over the `d` axis, then scale by `weight` and add the
/// optional `bias`.
///
/// `x`, `y` are column-major `(d, cols)`, so column `c` is the contiguous slice
/// `c*d .. (c+1)*d`. `weight` (and `bias`) have length `d`. This fuses the
/// transpose→norm→transpose sequence the eager path otherwise runs for a
/// feature-first activation into one pass, matching `layer_norm_host`.
pub fn layer_norm_feature_first_into(
    x: &[f32],
    d: usize,
    cols: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    eps: f32,
    y: &mut [f32],
) {
    debug_assert_eq!(x.len(), d * cols);
    debug_assert_eq!(y.len(), d * cols);
    debug_assert_eq!(weight.len(), d);
    if d == 0 || cols == 0 {
        return;
    }
    let denom = d as f32;
    let normalize = |out: &mut [f32], input: &[f32]| {
        let mut mean = 0.0f32;
        for value in input {
            mean += *value;
        }
        mean /= denom;
        let mut var = 0.0f32;
        for value in input {
            let centered = *value - mean;
            var += centered * centered;
        }
        var /= denom;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..d {
            let mut value = (input[i] - mean) * inv * weight[i];
            if let Some(bias) = bias {
                value += bias[i];
            }
            out[i] = value;
        }
    };
    if cols > 1 && d.saturating_mul(cols) >= PARALLEL_THRESHOLD {
        y.par_chunks_mut(d)
            .zip(x.par_chunks(d))
            .for_each(|(out, input)| normalize(out, input));
    } else {
        for c in 0..cols {
            normalize(&mut y[c * d..(c + 1) * d], &x[c * d..(c + 1) * d]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(weight: &[f32], in_dim: usize, out_dim: usize, x: &[f32], length: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; out_dim * length];
        for o in 0..out_dim {
            for t in 0..length {
                let mut acc = 0.0f32;
                for i in 0..in_dim {
                    acc += weight[i * out_dim + o] * x[i * length + t];
                }
                y[o * length + t] = acc;
            }
        }
        y
    }

    fn lcg(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*state >> 40) as f32) / (1u64 << 24) as f32 - 0.5
    }

    #[test]
    fn matches_naive_loop_small() {
        let mut state = 12345u64;
        for (in_dim, out_dim, length) in [(4usize, 3usize, 5usize), (8, 8, 8), (1, 2, 3), (3, 5, 1)]
        {
            let weight: Vec<f32> = (0..in_dim * out_dim).map(|_| lcg(&mut state)).collect();
            let x: Vec<f32> = (0..in_dim * length).map(|_| lcg(&mut state)).collect();
            let got = matmul_row_major(&weight, in_dim, out_dim, &x, length);
            let want = naive(&weight, in_dim, out_dim, &x, length);
            let diff = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(diff < 1e-6, "diff {diff} at {in_dim}x{out_dim}x{length}");
        }
    }

    #[test]
    fn matches_naive_loop_parallel_path() {
        // Large enough to cross PARALLEL_THRESHOLD and exercise the row blocks.
        let (in_dim, out_dim, length) = (256usize, 300usize, 64usize);
        let mut state = 99u64;
        let weight: Vec<f32> = (0..in_dim * out_dim).map(|_| lcg(&mut state)).collect();
        let x: Vec<f32> = (0..in_dim * length).map(|_| lcg(&mut state)).collect();
        assert!(in_dim * out_dim * length >= PARALLEL_THRESHOLD);
        let got = matmul_row_major(&weight, in_dim, out_dim, &x, length);
        let want = naive(&weight, in_dim, out_dim, &x, length);
        let diff = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff < 1e-3, "diff {diff}");
    }

    #[test]
    fn add_accumulates() {
        let weight = [1.0f32, 2.0, 3.0, 4.0]; // (in=2, out=2)
        let x = [1.0f32, 0.0, 0.0, 1.0]; // (in=2, length=2)
        let mut y = [10.0f32; 4];
        matmul_row_major_add_into(&weight, 2, 2, &x, 2, &mut y);
        // weightᵀ x = [[1, 3], [2, 4]] (row-major (out=2, length=2))
        assert_eq!(y, [11.0, 13.0, 12.0, 14.0]);
    }

    #[test]
    fn input_mul_weight_transpose_matches_naive() {
        // weight is row-major (out=2, in=2): W[0] = [1, 2], W[1] = [3, 4].
        let weight = [1.0f32, 2.0, 3.0, 4.0];
        let x = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]; // (rows=3, in=2)
        let got = input_mul_weight_transpose(&x, 3, 2, &weight, 2);
        // rows: [1, 3], [2, 4], [3, 7]
        assert_eq!(got, [1.0, 3.0, 2.0, 4.0, 3.0, 7.0]);
    }

    #[test]
    fn input_mul_weight_transpose_parallel_path() {
        let (rows, in_dim, out_dim) = (300usize, 256, 64);
        let mut state = 7u64;
        let x: Vec<f32> = (0..rows * in_dim).map(|_| lcg(&mut state)).collect();
        let weight: Vec<f32> = (0..out_dim * in_dim).map(|_| lcg(&mut state)).collect();
        assert!(rows * in_dim * out_dim >= PARALLEL_THRESHOLD);
        let got = input_mul_weight_transpose(&x, rows, in_dim, &weight, out_dim);
        let mut want = vec![0.0f32; rows * out_dim];
        for row in 0..rows {
            for o in 0..out_dim {
                let mut acc = 0.0f32;
                for i in 0..in_dim {
                    acc += weight[o * in_dim + i] * x[row * in_dim + i];
                }
                want[row * out_dim + o] = acc;
            }
        }
        let diff = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff < 1e-3, "diff {diff}");
    }
}
