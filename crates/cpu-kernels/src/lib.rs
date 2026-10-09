//! Host CPU kernels shared by the engines.
//!
//! These run on plain `&[f32]` host slices and have no tenferro dependency, so
//! both the host forwards and the tenferro-backed paths can share one
//! implementation. The matmul kernels are BLAS-class (`matrixmultiply`) and
//! parallelized across output rows with `rayon` when the work is large enough.
//!
//! Layouts follow the engines' convention: weights are row-major `(in, out)`
//! and activations are row-major `(dim, length)`.

#![allow(clippy::approx_constant, clippy::excessive_precision)]

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
    assert_eq!(x.len(), rows * in_dim);
    assert_eq!(weight.len(), out_dim * in_dim);
    assert_eq!(y.len(), rows * out_dim);
    if rows == 0 || out_dim == 0 || in_dim == 0 {
        return;
    }
    // y ← x · weightᵀ + y. `weightᵀ` is `(in_dim, out_dim)` with row stride 1
    // (the `in_dim` axis of the row-major `(out_dim, in_dim)` storage).
    #[cfg(feature = "openblas")]
    if rows >= 32
        && [rows, in_dim, out_dim]
            .iter()
            .all(|&n| n <= i32::MAX as usize)
    {
        #[link(name = "openblas")]
        unsafe extern "C" {
            fn cblas_sgemm(
                layout: i32,
                trans_a: i32,
                trans_b: i32,
                m: i32,
                n: i32,
                k: i32,
                alpha: f32,
                a: *const f32,
                lda: i32,
                b: *const f32,
                ldb: i32,
                beta: f32,
                c: *mut f32,
                ldc: i32,
            );
        }
        // SAFETY: nonempty dimensions fit the LP64 CBLAS integer ABI; the
        // validated slices describe row-major X, W and accumulated Y. The
        // output is exclusively borrowed and all leading dimensions match.
        unsafe {
            cblas_sgemm(
                101,
                111,
                112,
                rows as i32,
                out_dim as i32,
                in_dim as i32,
                1.0,
                x.as_ptr(),
                in_dim as i32,
                weight.as_ptr(),
                in_dim as i32,
                1.0,
                y.as_mut_ptr(),
                out_dim as i32,
            );
        }
        return;
    }
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

/// `y = x · weight` for **column-major** `x (length, in_dim)` and
/// `y (length, out_dim)`, with `weight` stored **row-major `(in_dim, out_dim)`**
/// (the natural safetensors `Vec<f32>`).
///
/// `matrixmultiply` favors a column-major `A` (`rsa = 1`) and a row-major `B`
/// (`csb = 1`); the column-major activations and the raw row-major weight are
/// exactly that. So `A = x` (`rsa = 1`, `csa = length`), `B = weight`
/// (`rsb = out_dim`, `csb = 1`), `C = y` (`rsc = 1`, `csc = length`), and the
/// contraction over `i` gives `y[t, o] = Σᵢ x[t, i] · weight[i, o]` with no
/// transpose and the fast `matrixmultiply` path.
pub fn matmul_col_major_into(
    x: &[f32],
    weight: &[f32],
    length: usize,
    in_dim: usize,
    out_dim: usize,
    y: &mut [f32],
) {
    debug_assert_eq!(x.len(), length * in_dim);
    debug_assert_eq!(weight.len(), in_dim * out_dim);
    debug_assert_eq!(y.len(), length * out_dim);
    if length == 0 || in_dim == 0 || out_dim == 0 {
        return;
    }
    let work = length.saturating_mul(in_dim).saturating_mul(out_dim);
    if work >= PARALLEL_THRESHOLD && out_dim > 1 {
        // Block over the output features so each rayon task owns a disjoint
        // column block of the column-major `y` (o stride = length).
        let threads = rayon::current_num_threads().max(1);
        let block = out_dim.div_ceil(threads * 4).max(8).min(out_dim);
        let nblocks = out_dim.div_ceil(block);
        let x_addr = x.as_ptr() as usize;
        let w_addr = weight.as_ptr() as usize;
        let y_addr = y.as_mut_ptr() as usize;
        (0..nblocks).into_par_iter().for_each(|bi| {
            let o0 = bi * block;
            let n = (out_dim - o0).min(block);
            // SAFETY: each task writes the disjoint output columns `[o0, o0+n)`
            // of every `y` row and reads only `x`/`weight`; the raw addresses
            // recapture the slices rayon cannot borrow for the strided output.
            unsafe {
                let x_ptr = x_addr as *const f32;
                let w_ptr = (w_addr as *const f32).add(o0);
                let y_ptr = (y_addr as *mut f32).add(length * o0);
                matrixmultiply::sgemm(
                    length,
                    in_dim,
                    n,
                    1.0,
                    x_ptr,
                    1,
                    length as isize,
                    w_ptr,
                    out_dim as isize,
                    1,
                    0.0,
                    y_ptr,
                    1,
                    length as isize,
                );
            }
        });
    } else {
        // SAFETY: the slices are validated by the debug assertions; the strides
        // describe exactly those shapes.
        unsafe {
            matrixmultiply::sgemm(
                length,
                in_dim,
                out_dim,
                1.0,
                x.as_ptr(),
                1,
                length as isize,
                weight.as_ptr(),
                out_dim as isize,
                1,
                0.0,
                y.as_mut_ptr(),
                1,
                length as isize,
            );
        }
    }
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

/// `erf` as evaluated by the MLX Metal kernel (`mathfns.jl` `mlx_erf`), `f32`.
///
/// Shared by the `erf` extension op and the fused GeGLU kernel so both match
/// Laya's reference.
pub fn erf_f32(a: f32) -> f32 {
    let t = a.abs();
    let s = a * a;
    if t > 0.927_734_4 {
        let mut r = (-1.728_534_7e-5f32).mul_add(t, 3.831_971_3e-4);
        let u = (-3.883_964_4e-3f32).mul_add(t, 2.425_462_2e-2);
        r = r.mul_add(s, u);
        r = r.mul_add(t, -1.067_778_8e-1);
        r = r.mul_add(t, -6.348_466_9e-1);
        r = r.mul_add(t, -1.287_175_1e-1);
        r = r.mul_add(t, -t);
        r = -mlx_expm1f(r);
        r.copysign(a)
    } else {
        let mut r = -5.967_617e-4f32;
        r = r.mul_add(s, 4.991_194_2e-3);
        r = r.mul_add(s, -2.676_813_5e-2);
        r = r.mul_add(s, 1.128_199_2e-1);
        r = r.mul_add(s, -3.761_253_4e-1);
        r = r.mul_add(s, 1.283_791_7e-1);
        r.mul_add(a, a)
    }
}

/// `expm1` as evaluated by the MLX Metal kernel (`mathfns.jl` `mlx_expm1f`).
fn mlx_expm1f(a: f32) -> f32 {
    let mut j = 1.442695f32.mul_add(a, 12582912.0);
    j -= 12582912.0;
    let i = j as i32;
    let f = j.mul_add(-0.693145_752f32, a);
    let s = if a == 0.0 { a } else { f * f };
    let mut r = 1.973_509_8e-4f32;
    r = r.mul_add(f, 1.393_090_7e-3);
    r = r.mul_add(f, 8.333_44e-3);
    r = r.mul_add(f, 4.166_680_2e-2);
    r = r.mul_add(f, 1.666_667_2e-1);
    r = r.mul_add(f, 4.999_999_7e-1);
    let u = if j == 1.0 { f + 0.5 } else { f };
    let v = r.mul_add(s, u);
    let half = 0.5f32;
    let t = half * 2.0f32.powi(i);
    let y = t - half;
    let x = (t - y) - half;
    r = v.mul_add(t, x) + y;
    r += r;
    if j == 0.0 {
        r = v;
    }
    if j == 1.0 {
        r = v + v;
    }
    if (a - 1.0).abs() > 88.0 {
        let e = a.exp2();
        r = e.mul_add(e, -1.0);
    }
    r
}

/// Exact (erf-based) GELU: `x * (1 + erf(x / sqrt(2))) / 2`.
#[inline]
pub fn gelu_erf_f32(x: f32) -> f32 {
    0.5 * x * (1.0 + erf_f32(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// GeGLU in Laya's order: `gelu(value) * gate`, for column-major
/// `u (2*intermediate, cols)` → `out (intermediate, cols)`. The `value` half is
/// channels `[0, intermediate)`, the `gate` half `[intermediate, 2*intermediate)`.
pub fn geglu_into(u: &[f32], intermediate: usize, cols: usize, out: &mut [f32]) {
    debug_assert_eq!(u.len(), 2 * intermediate * cols);
    debug_assert_eq!(out.len(), intermediate * cols);
    if intermediate == 0 || cols == 0 {
        return;
    }
    let activate = |input: &[f32], output: &mut [f32]| {
        for i in 0..intermediate {
            output[i] = gelu_erf_f32(input[i]) * input[intermediate + i];
        }
    };
    // The erf polynomial performs many fused multiply-adds per element, so
    // it benefits from threading at smaller sizes than a dense projection.
    if cols > 1 && intermediate.saturating_mul(cols) >= 1 << 14 {
        out.par_chunks_mut(intermediate)
            .zip(u.par_chunks(2 * intermediate))
            .for_each(|(output, input)| activate(input, output));
    } else {
        for (output, input) in out.chunks_mut(intermediate).zip(u.chunks(2 * intermediate)) {
            activate(input, output);
        }
    }
}

/// Feature-last RMSNorm over a column-major `(rows, width)` activation: for
/// each row `r`, scale by `1 / sqrt(mean(x²) + eps)` and by
/// `centered ? 1 + weight[d] : weight[d]`. Mirrors `jeff_infer`'s
/// `norm::rms_norm` on a `(length, hidden)` activation (the norm axis is the
/// last one, with stride `rows`).
pub fn rms_norm_last_into(
    x: &[f32],
    rows: usize,
    width: usize,
    weight: &[f32],
    centered: bool,
    eps: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), rows * width);
    debug_assert_eq!(out.len(), rows * width);
    debug_assert_eq!(weight.len(), width);
    if rows == 0 || width == 0 {
        return;
    }
    let inv_width = 1.0 / (width as f32);
    let x_addr = x.as_ptr() as usize;
    let out_addr = out.as_mut_ptr() as usize;
    (0..rows).into_par_iter().for_each(|r| {
        // SAFETY: each task owns row `r` (indices `r + rows*d`), so writes are
        // disjoint and `x`/`weight` are only read.
        unsafe {
            let x = x_addr as *const f32;
            let out = out_addr as *mut f32;
            let mut mean_sq = 0.0f32;
            for d in 0..width {
                let v = *x.add(r + rows * d);
                mean_sq += v * v;
            }
            let scale = 1.0 / (mean_sq * inv_width + eps).sqrt();
            for (d, &w) in weight.iter().enumerate() {
                let wt = if centered { 1.0 + w } else { w };
                *out.add(r + rows * d) = *x.add(r + rows * d) * scale * wt;
            }
        }
    });
}

/// `silu(gate) * up` elementwise (`silu(x) = x / (1 + exp(-x))`).
pub fn gated_silu_into(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(out.len(), gate.len());
    let g_addr = gate.as_ptr() as usize;
    let u_addr = up.as_ptr() as usize;
    let o_addr = out.as_mut_ptr() as usize;
    let per = 4096usize;
    let chunks = out.len().div_ceil(per);
    (0..chunks).into_par_iter().for_each(|c| {
        let start = c * per;
        let end = (start + per).min(out.len());
        // SAFETY: each task owns the disjoint output range `[start, end)`; `gate`
        // and `up` are only read.
        unsafe {
            let gate = g_addr as *const f32;
            let up = u_addr as *const f32;
            let out = o_addr as *mut f32;
            for i in start..end {
                let g = *gate.add(i);
                *out.add(i) = (g / (1.0 + (-g).exp())) * *up.add(i);
            }
        }
    });
}

/// projections (`width = heads * head_dim`, column-major), returning the
/// merged `(length, width)` activation ready for the output projection.
///
/// Per head: centered RMSNorm over `head_dim` (`scale = 1 + weight`), partial
/// RoPE on the first `rotary_dim` channels (pairs `(i, i + rotary_dim/2)`,
/// angle `t * theta^(-2i/rotary_dim)`), causal + active-key masked attention,
/// then a sigmoid query gate. Mirrors `full_attention_host`; `out` is written in
/// the contract-last `(length, width)` column-major layout.
#[allow(clippy::too_many_arguments)]
pub fn jeff_full_attention_into(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    mask: &[f32],
    length: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
    eps: f32,
    out: &mut [f32],
) {
    let width = heads * head_dim;
    debug_assert_eq!(q.len(), length * width);
    debug_assert_eq!(k.len(), length * width);
    debug_assert_eq!(v.len(), length * width);
    debug_assert_eq!(gate.len(), length * width);
    debug_assert_eq!(q_norm.len(), head_dim);
    debug_assert_eq!(k_norm.len(), head_dim);
    debug_assert_eq!(mask.len(), length);
    debug_assert_eq!(out.len(), length * width);
    if length == 0 || width == 0 || head_dim == 0 {
        return;
    }
    let attn_scale = 1.0 / (head_dim as f32).sqrt();
    let half = rotary_dim / 2;
    let q_addr = q.as_ptr() as usize;
    let k_addr = k.as_ptr() as usize;
    let v_addr = v.as_ptr() as usize;
    let gate_addr = gate.as_ptr() as usize;
    let out_addr = out.as_mut_ptr() as usize;
    (0..heads).into_par_iter().for_each(|head| {
        // SAFETY: each task owns the feature band `[head*head_dim,
        // (head+1)*head_dim)` of `out` for every length, so writes are disjoint;
        // q/k/v/gate/mask are only read.
        let feat = head * head_dim;
        unsafe {
            let q = q_addr as *const f32;
            let k = k_addr as *const f32;
            let v = v_addr as *const f32;
            let gate = gate_addr as *const f32;
            let out = out_addr as *mut f32;
            let mut qh = vec![0.0f32; length * head_dim];
            let mut kh = vec![0.0f32; length * head_dim];
            let mut vh = vec![0.0f32; length * head_dim];
            for t in 0..length {
                let src = t + length * feat;
                let dst = t * head_dim;
                for i in 0..head_dim {
                    qh[dst + i] = *q.add(src + length * i);
                    kh[dst + i] = *k.add(src + length * i);
                    vh[dst + i] = *v.add(src + length * i);
                }
            }
            let norm = |data: &mut [f32], weight: &[f32]| {
                for t in 0..length {
                    let row = t * head_dim;
                    let mut mean_sq = 0.0f32;
                    for i in 0..head_dim {
                        mean_sq += data[row + i] * data[row + i];
                    }
                    mean_sq /= head_dim as f32;
                    let scale = 1.0 / (mean_sq + eps).sqrt();
                    for i in 0..head_dim {
                        data[row + i] *= scale * (1.0 + weight[i]);
                    }
                }
            };
            norm(&mut qh, q_norm);
            norm(&mut kh, k_norm);
            if half > 0 {
                for t in 0..length {
                    let row = t * head_dim;
                    for i in 0..half {
                        let angle =
                            (t as f32) * theta.powf(-2.0 * (i as f32) / (rotary_dim as f32));
                        let (c, s) = (angle.cos(), angle.sin());
                        let a = qh[row + i];
                        let b = qh[row + i + half];
                        qh[row + i] = a * c - b * s;
                        qh[row + i + half] = a * s + b * c;
                        let a = kh[row + i];
                        let b = kh[row + i + half];
                        kh[row + i] = a * c - b * s;
                        kh[row + i + half] = a * s + b * c;
                    }
                }
            }
            let mut probs = vec![0.0f32; length];
            let mut oh = vec![0.0f32; head_dim];
            for query in 0..length {
                let qo = query * head_dim;
                let mut max = f32::NEG_INFINITY;
                for (key, prob) in probs.iter_mut().enumerate() {
                    if key > query || *mask.as_ptr().add(key) == 0.0 {
                        *prob = f32::NEG_INFINITY;
                        continue;
                    }
                    let ko = key * head_dim;
                    let mut acc = 0.0f32;
                    for i in 0..head_dim {
                        acc += qh[qo + i] * kh[ko + i];
                    }
                    let score = acc * attn_scale;
                    *prob = score;
                    if score > max {
                        max = score;
                    }
                }
                let mut sum = 0.0f32;
                for value in probs.iter_mut() {
                    *value = if value.is_finite() {
                        (*value - max).exp()
                    } else {
                        0.0
                    };
                    sum += *value;
                }
                for slot in oh.iter_mut() {
                    *slot = 0.0;
                }
                if sum > 0.0 {
                    for (key, &raw) in probs.iter().enumerate() {
                        let p = raw / sum;
                        if p == 0.0 {
                            continue;
                        }
                        let vo = key * head_dim;
                        for i in 0..head_dim {
                            oh[i] += vh[vo + i] * p;
                        }
                    }
                }
                for (i, &value) in oh.iter().enumerate() {
                    let g = *gate.add(query + length * (feat + i));
                    let sigmoid = 1.0 / (1.0 + (-g).exp());
                    *out.add(query + length * (feat + i)) = value * sigmoid;
                }
            }
        }
    });
}

/// Reusable attention head buffers and one RoPE table; no input/output data
/// escapes a forward. Use separate instances for concurrently running calls.
#[derive(Debug, Default)]
pub struct LayaAttentionWorkspace {
    heads: Vec<LayaHeadWorkspace>,
    rope_key: Option<(usize, usize, u32)>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}
#[derive(Debug, Default)]
struct LayaHeadWorkspace {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    probabilities: Vec<f32>,
    mixed: Vec<f32>,
}
impl LayaAttentionWorkspace {
    /// Owned buffer capacities, including unused head slots after shrinkage.
    pub fn retained_bytes(&self) -> usize {
        let floats = self
            .cos
            .capacity()
            .saturating_add(self.sin.capacity())
            .saturating_add(
                self.heads
                    .iter()
                    .map(|h| {
                        h.q.capacity()
                            .saturating_add(h.k.capacity())
                            .saturating_add(h.v.capacity())
                            .saturating_add(h.probabilities.capacity())
                            .saturating_add(h.mixed.capacity())
                    })
                    .sum::<usize>(),
            );
        floats.saturating_mul(size_of::<f32>()).saturating_add(
            self.heads
                .capacity()
                .saturating_mul(size_of::<LayaHeadWorkspace>()),
        )
    }
}

/// into `q`/`k`/`v`, apply ModernBERT RoPE to `q`/`k`, and run masked scaled
/// dot-product attention, returning `(d, length, batch)`.
///
/// Everything is done per `(batch, head)` in the host-friendly feature-first
/// layout: a head owns features `[head_dim*head, head_dim*(head+1))` with
/// `head_dim` contiguous (the host `attention_host` layout), so no transposes
/// are needed and the inner head_dim loop is vectorizable. `keep` is the
/// `(length, length, batch)` causal/window mask (`query + length*(key +
/// length*batch)`); `rope_base == 0.0` disables RoPE.
#[allow(clippy::too_many_arguments)]
pub fn laya_attention_block_into(
    qkv: &[f32],
    keep: &[bool],
    d: usize,
    heads: usize,
    length: usize,
    batch: usize,
    rope_base: f32,
    out: &mut [f32],
) {
    laya_attention_block_with_workspace(
        qkv,
        keep,
        d,
        heads,
        length,
        batch,
        rope_base,
        out,
        &mut LayaAttentionWorkspace::default(),
    );
}

/// Same computation as [`laya_attention_block_into`] with caller-owned scratch.
#[allow(clippy::too_many_arguments)]
pub fn laya_attention_block_with_workspace(
    qkv: &[f32],
    keep: &[bool],
    d: usize,
    heads: usize,
    length: usize,
    batch: usize,
    rope_base: f32,
    out: &mut [f32],
    workspace: &mut LayaAttentionWorkspace,
) {
    let n = d
        .checked_mul(length)
        .and_then(|n| n.checked_mul(batch))
        .expect("attention dimensions overflow");
    let qkv_size = n.checked_mul(3).expect("attention dimensions overflow");
    let mask_size = length
        .checked_mul(length)
        .and_then(|n| n.checked_mul(batch))
        .expect("attention mask dimensions overflow");
    assert_eq!(qkv.len(), qkv_size, "qkv length");
    assert_eq!(out.len(), n, "output length");
    assert_eq!(keep.len(), mask_size, "mask length");
    if n == 0 || heads == 0 || d % heads != 0 {
        return;
    }
    let hd = d / heads;
    let half = hd / 2;
    let scale = 1.0 / (hd as f32).sqrt();

    // RoPE tables `(length, half)`, built once and shared across heads.
    let use_rope = rope_base != 0.0;
    let rope_key = (length, hd, rope_base.to_bits());
    if use_rope && workspace.rope_key != Some(rope_key) {
        workspace.cos.resize(length * half, 0.);
        workspace.sin.resize(length * half, 0.);
        for l in 0..length {
            for i in 0..half {
                let theta = (l as f32) * rope_base.powf(-2. * (i as f32) / (hd as f32));
                workspace.cos[l * half + i] = theta.cos();
                workspace.sin[l * half + i] = theta.sin();
            }
        }
        workspace.rope_key = Some(rope_key);
    }

    // Share the union of each query tile's allowed key span across heads.
    // Sliding masks then avoid a dense contraction over unrelated keys.
    let tile = 64.min(length);
    let tiles = length.div_ceil(tile);
    let use_gemm = length >= 32;
    let key_ranges: Vec<(usize, usize, usize)> = if use_gemm {
        (0..batch)
            .flat_map(|b| {
                (0..length).step_by(tile).map(move |start| {
                    let end = (start + tile).min(length);
                    let mut first = length;
                    let mut last = 0;
                    let mut allowed = 0;
                    for key in 0..length {
                        let column = length * (key + length * b);
                        let active = keep[column + start..column + end]
                            .iter()
                            .filter(|value| **value)
                            .count();
                        if active > 0 {
                            first = first.min(key);
                            last = key + 1;
                            allowed += active;
                        }
                    }
                    if first == length {
                        (0, 0, 0)
                    } else {
                        (first, last, allowed)
                    }
                })
            })
            .collect()
    } else {
        Vec::new()
    };
    let max_keys = key_ranges
        .iter()
        .map(|(first, last, _)| last - first)
        .max()
        .unwrap_or(0);

    let dense_batches: Vec<bool> = if use_gemm {
        key_ranges
            .chunks(tiles)
            .map(|ranges| {
                ranges
                    .iter()
                    .enumerate()
                    .all(|(index, (first, last, allowed))| {
                        let queries = (length - index * tile).min(tile);
                        *allowed >= (queries * (last - first)).div_ceil(2)
                    })
            })
            .collect()
    } else {
        Vec::new()
    };

    let qkv_addr = qkv.as_ptr() as usize;
    let keep_addr = keep.as_ptr() as usize;
    let out_addr = out.as_mut_ptr() as usize;
    let cos_addr = workspace.cos.as_ptr() as usize;
    let sin_addr = workspace.sin.as_ptr() as usize;
    workspace.heads.resize_with(
        workspace.heads.len().max(batch * heads),
        LayaHeadWorkspace::default,
    );
    workspace.heads[..batch * heads]
        .par_iter_mut()
        .enumerate()
        .for_each(|(bh, scratch)| {
            // SAFETY: each task owns one `(batch, head)` pair — a disjoint feature
            // band of `out` — and only reads `qkv`/`keep`/`cos`/`sin`.
            let b = bh / heads;
            let head = bh % heads;
            let feat = head * hd;
            unsafe {
                let qkv = qkv_addr as *const f32;
                let keep = keep_addr as *const bool;
                let out = out_addr as *mut f32;
                let LayaHeadWorkspace {
                    q,
                    k,
                    v,
                    probabilities,
                    mixed,
                } = scratch;
                q.resize(length * hd, 0.);
                k.resize(length * hd, 0.);
                v.resize(length * hd, 0.);
                // split qkv -> q/k/v head band
                for l in 0..length {
                    let src = 3 * d * (l + length * b) + feat;
                    let dst = l * hd;
                    for i in 0..hd {
                        q[dst + i] = *qkv.add(src + i);
                        k[dst + i] = *qkv.add(src + d + i);
                        v[dst + i] = *qkv.add(src + 2 * d + i);
                    }
                }
                if use_rope {
                    let cos = cos_addr as *const f32;
                    let sin = sin_addr as *const f32;
                    for l in 0..length {
                        let row = l * hd;
                        let table = l * half;
                        for i in 0..half {
                            let c = *cos.add(table + i);
                            let s = *sin.add(table + i);
                            let a = q[row + i];
                            let bb = q[row + i + half];
                            q[row + i] = a * c - bb * s;
                            q[row + i + half] = a * s + bb * c;
                            let a = k[row + i];
                            let bb = k[row + i + half];
                            k[row + i] = a * c - bb * s;
                            k[row + i + half] = a * s + bb * c;
                        }
                    }
                }
                // For longer sequences, use library GEMM for QKᵀ and PV. Query
                // tiles bound scratch at O(64 * length), rather than length².
                // A nonfinite V needs the scalar path: masked zero probabilities
                // must not turn excluded NaNs/infinities into 0 * NaN in dense PV.
                if use_gemm && dense_batches[b] && v.iter().all(|value| value.is_finite()) {
                    let score_size = tile
                        .checked_mul(max_keys)
                        .expect("attention workspace overflow");
                    probabilities.resize(score_size, 0.);
                    mixed.resize(tile * hd, 0.);
                    for (index, start) in (0..length).step_by(tile).enumerate() {
                        let queries = (length - start).min(tile);
                        let (key_start, key_end, _) = key_ranges[b * tiles + index];
                        let keys = key_end - key_start;
                        if keys == 0 {
                            // Match the scalar path's all-masked softmax semantics.
                            for local in 0..queries {
                                let destination = feat + d * (start + local + length * b);
                                for i in 0..hd {
                                    *out.add(destination + i) = f32::NAN;
                                }
                            }
                            continue;
                        }
                        // The outer Rayon task owns this head; these SGEMMs are
                        // single-threaded, with disjoint owned scratch matrices.
                        matrixmultiply::sgemm(
                            queries,
                            hd,
                            keys,
                            1.0,
                            q.as_ptr().add(start * hd),
                            hd as isize,
                            1,
                            k.as_ptr().add(key_start * hd),
                            1,
                            hd as isize,
                            0.0,
                            probabilities.as_mut_ptr(),
                            keys as isize,
                            1,
                        );
                        for local in 0..queries {
                            let query = start + local;
                            let row = &mut probabilities[local * keys..(local + 1) * keys];
                            let mut max = f32::NEG_INFINITY;
                            for (key, value) in row.iter_mut().enumerate() {
                                *value =
                                    if *keep.add(query + length * (key_start + key + length * b)) {
                                        *value * scale
                                    } else {
                                        f32::NEG_INFINITY
                                    };
                                if *value > max {
                                    max = *value;
                                }
                            }
                            let mut sum = 0.0f32;
                            for value in row.iter_mut() {
                                *value = if value.is_finite() {
                                    (*value - max).exp()
                                } else {
                                    0.0
                                };
                                sum += *value;
                            }
                            for value in row {
                                *value /= sum;
                            }
                        }
                        matrixmultiply::sgemm(
                            queries,
                            keys,
                            hd,
                            1.0,
                            probabilities.as_ptr(),
                            keys as isize,
                            1,
                            v.as_ptr().add(key_start * hd),
                            hd as isize,
                            1,
                            0.0,
                            mixed.as_mut_ptr(),
                            hd as isize,
                            1,
                        );
                        for local in 0..queries {
                            let destination = feat + d * (start + local + length * b);
                            std::ptr::copy_nonoverlapping(
                                mixed.as_ptr().add(local * hd),
                                out.add(destination),
                                hd,
                            );
                        }
                    }
                    return;
                }
                // masked scaled dot-product attention for this head
                probabilities.resize(length, 0.);
                let probs = probabilities;
                for query in 0..length {
                    let qo = query * hd;
                    let mut max = f32::NEG_INFINITY;
                    for (key, prob) in probs.iter_mut().enumerate() {
                        if !*keep.add(query + length * (key + length * b)) {
                            *prob = f32::NEG_INFINITY;
                            continue;
                        }
                        let ko = key * hd;
                        let mut acc = 0.0f32;
                        for i in 0..hd {
                            acc += q[qo + i] * k[ko + i];
                        }
                        let score = acc * scale;
                        *prob = score;
                        if score > max {
                            max = score;
                        }
                    }
                    let mut sum = 0.0f32;
                    for value in probs.iter_mut() {
                        *value = if value.is_finite() {
                            (*value - max).exp()
                        } else {
                            0.0
                        };
                        sum += *value;
                    }
                    let o_base = feat + d * (query + length * b);
                    for i in 0..hd {
                        *out.add(o_base + i) = 0.0;
                    }
                    for (key, &raw) in probs.iter().enumerate() {
                        let prob = raw / sum;
                        if prob == 0.0 {
                            continue;
                        }
                        let vo = key * hd;
                        for i in 0..hd {
                            *out.add(o_base + i) += v[vo + i] * prob;
                        }
                    }
                }
            }
        });
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

    #[cfg(feature = "openblas")]
    #[test]
    fn openblas_projection_preserves_layout_and_accumulation() {
        for (rows, in_dim, out_dim) in [(32, 67, 37), (65, 33, 1)] {
            let x: Vec<f32> = (0..rows * in_dim)
                .map(|i| (i % 17) as f32 / 17.0 - 0.5)
                .collect();
            let weight: Vec<f32> = (0..out_dim * in_dim)
                .map(|i| (i % 23) as f32 / 23.0 - 0.5)
                .collect();
            let mut output = vec![0.25; rows * out_dim];
            input_mul_weight_transpose_add_into(&x, rows, in_dim, &weight, out_dim, &mut output);
            for row in 0..rows {
                for column in 0..out_dim {
                    let expected = 0.25
                        + (0..in_dim)
                            .map(|i| x[row * in_dim + i] * weight[column * in_dim + i])
                            .sum::<f32>();
                    assert!((output[row * out_dim + column] - expected).abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "qkv length")]
    fn laya_attention_rejects_short_storage_before_raw_access() {
        let mut output = vec![0.0f32; 4 * 32];
        laya_attention_block_into(&[], &vec![true; 32 * 32], 4, 1, 32, 1, 0.0, &mut output);
    }

    #[test]
    fn laya_attention_preserves_masked_nonfinite_values_and_empty_rows() {
        let length = 65;
        let d = 4;
        let mut qkv = vec![0.0f32; 3 * d * length];
        for key in 0..length {
            for feature in 0..d {
                qkv[3 * d * key + 2 * d + feature] = (feature + 1) as f32;
            }
        }
        let mut keep = vec![true; length * length];
        for query in 0..length {
            keep[query + length * 32] = false;
        }
        for key in 0..length {
            keep[length * key] = false;
            keep[length - 1 + length * key] = false;
        }
        for excluded in [1.0f32, f32::NAN, f32::INFINITY] {
            for feature in 0..d {
                qkv[3 * d * 32 + 2 * d + feature] = excluded;
            }
            let mut output = vec![0.0; d * length];
            laya_attention_block_into(&qkv, &keep, d, 1, length, 1, 0.0, &mut output);
            for query in 0..length {
                for feature in 0..d {
                    let actual = output[query * d + feature];
                    if query == 0 || query == length - 1 {
                        assert!(actual.is_nan());
                    } else {
                        assert!(
                            (actual - (feature + 1) as f32).abs() <= 1e-6,
                            "query={query}, feature={feature}, actual={actual}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn laya_attention_workspace_reuses_only_scratch_across_inputs() {
        let mut workspace = LayaAttentionWorkspace::default();
        let mut state = 654321u64;
        let mut retained = 0;
        for (length, batch, heads, base) in [
            (8, 8, 2, 10000.0),
            (65, 2, 2, 10000.0),
            (3, 1, 1, 160000.0),
            (32, 3, 4, 0.0),
            (8, 8, 2, 10000.0),
            (65, 2, 2, 160000.0),
        ] {
            let d = 8;
            let qkv: Vec<f32> = (0..3 * d * length * batch)
                .map(|_| lcg(&mut state))
                .collect();
            let keep: Vec<bool> = (0..length * length * batch).map(|i| i % 7 != 0).collect();
            let mut expected = vec![0.0; d * length * batch];
            let mut actual = vec![f32::NAN; expected.len()];
            laya_attention_block_into(&qkv, &keep, d, heads, length, batch, base, &mut expected);
            laya_attention_block_with_workspace(
                &qkv,
                &keep,
                d,
                heads,
                length,
                batch,
                base,
                &mut actual,
                &mut workspace,
            );
            for (a, e) in actual.iter().zip(&expected) {
                assert!(a.to_bits() == e.to_bits() || (a.is_nan() && e.is_nan()));
            }
            assert!(workspace.retained_bytes() >= retained);
            retained = workspace.retained_bytes();
        }
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

    #[test]
    fn threaded_geglu_preserves_scalar_results_and_column_layout() {
        // Uneven feature/column counts exercise partial Rayon work splits.
        let (intermediate, cols) = (257, 65);
        let input: Vec<f32> = (0..2 * intermediate * cols)
            .map(|i| ((i % 103) as f32 - 51.0) / 7.0)
            .collect();
        let expected: Vec<f32> = input
            .chunks_exact(2 * intermediate)
            .flat_map(|column| {
                column[..intermediate]
                    .iter()
                    .zip(&column[intermediate..])
                    .map(|(&value, &gate)| gelu_erf_f32(value) * gate)
            })
            .collect();
        for threads in [1, 3] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut output = vec![f32::NAN; intermediate * cols];
            pool.install(|| geglu_into(&input, intermediate, cols, &mut output));
            assert_eq!(
                output.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}

#[cfg(feature = "onednn")]
pub mod prepared_projection;
