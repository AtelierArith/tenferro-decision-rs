//! Experimental `#[cube]` (CubeCL) versions of the raw CUDA path's non-GEMM
//! kernels (`cuda` feature; selected with `TENFERRO_DECISION_CUDA_KERNELS=cubecl`).
//!
//! These are portable CubeCL kernels (the same source can target CubeCL's
//! wgpu/Metal runtimes) launched on tenferro's own CubeCL client from inside
//! the raw scope; GEMMs stay on cuBLAS. Buffers are tenferro pool tensors
//! (CubeCL cannot bind external device pointers), addressed by element
//! offsets. Layouts and semantics match the NVRTC kernels of the raw path
//! (`laya_raw.cu`, `jeff_raw.cu`, `delta_raw.cu`).

// Kernel bodies keep nested `if`s (a `&&` guard is not short-circuit
// evaluated in `#[cube]` code, so an index behind it could be read out of
// bounds) and spell out `(a + b - 1) / b` (plain integer ops in every
// CubeCL backend).
#![allow(clippy::collapsible_if, clippy::manual_div_ceil)]

use cubecl::prelude::*;

/// Units per cube of the plane-per-column norms; a cube normalizes
/// `NORM_UNITS / PLANE_DIM` columns (8 on CUDA's 32-lane planes).
pub const NORM_UNITS: u32 = 256;

/// Launch geometry of the plane-per-column norms for `cols` columns. Uses
/// the device's largest plane size, so every column is covered whatever
/// plane size the kernel actually runs with.
pub fn norm_dims<R: Runtime>(client: &ComputeClient<R>, cols: u32) -> (CubeCount, CubeDim) {
    let plane = client.properties().hardware.plane_size_max.max(1);
    let per_cube = (NORM_UNITS / plane).max(1);
    (
        CubeCount::Static(cols.div_ceil(per_cube), 1, 1),
        CubeDim::new_1d(NORM_UNITS),
    )
}

/// LayerNorm of `ncol` columns of length `d` at `x[xo..]` into `y[yo..]`
/// (`y` may alias `x`): `(x - mean) / sqrt(var + eps) * w (+ b)`. One plane
/// per column (any plane size), `NORM_UNITS / PLANE_DIM` columns per cube;
/// the column is re-read from memory (cache-resident) instead of being kept
/// in a fixed register array, so no warp width is assumed.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn cube_layernorm(
    y: &mut Array<f32>,
    x: &Array<f32>,
    w: &Array<f32>,
    b: &Array<f32>,
    yo: u32,
    xo: u32,
    wo: u32,
    bo: u32,
    d: u32,
    ncol: u32,
    eps: f32,
    #[comptime] has_bias: bool,
) {
    let pd = PLANE_DIM;
    let lane = UNIT_POS_PLANE;
    let col = CUBE_POS_X * (CUBE_DIM_X / pd) + UNIT_POS_X / pd;
    if col < ncol {
        let src = xo + col * d;
        let dst = yo + col * d;
        let mut s = 0.0f32;
        let mut i = lane;
        while i < d {
            s += x[(src + i) as usize];
            i += pd;
        }
        let mu = plane_sum(s) / f32::cast_from(d);
        let mut q = 0.0f32;
        let mut i = lane;
        while i < d {
            let c = x[(src + i) as usize] - mu;
            q += c * c;
            i += pd;
        }
        let inv = 1.0f32 / f32::sqrt(plane_sum(q) / f32::cast_from(d) + eps);
        let mut i = lane;
        while i < d {
            let mut t = (x[(src + i) as usize] - mu) * inv * w[(wo + i) as usize];
            if has_bias {
                t += b[(bo + i) as usize];
            }
            y[(dst + i) as usize] = t;
            i += pd;
        }
    }
}

/// Centered RMSNorm `x * rsqrt(mean(x^2) + eps) * (1 + w)` of `ncol`
/// columns of length `d`, times `mask[mo + col]` when `masked`. Plane per
/// column, as [`cube_layernorm`].
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn cube_rms_centered(
    y: &mut Array<f32>,
    x: &Array<f32>,
    w: &Array<f32>,
    mask: &Array<f32>,
    yo: u32,
    xo: u32,
    wo: u32,
    mo: u32,
    d: u32,
    ncol: u32,
    eps: f32,
    #[comptime] masked: bool,
) {
    let pd = PLANE_DIM;
    let lane = UNIT_POS_PLANE;
    let col = CUBE_POS_X * (CUBE_DIM_X / pd) + UNIT_POS_X / pd;
    if col < ncol {
        let src = xo + col * d;
        let dst = yo + col * d;
        let mut s = 0.0f32;
        let mut i = lane;
        while i < d {
            let v = x[(src + i) as usize];
            s += v * v;
            i += pd;
        }
        let mut scale = 1.0f32 / f32::sqrt(plane_sum(s) / f32::cast_from(d) + eps);
        if masked {
            scale *= mask[(mo + col) as usize];
        }
        let mut i = lane;
        while i < d {
            y[(dst + i) as usize] = x[(src + i) as usize] * scale * (1.0f32 + w[(wo + i) as usize]);
            i += pd;
        }
    }
}

/// `expm1` as evaluated by the MLX Metal kernel (`mathfns.jl`).
#[cube]
fn mlx_expm1f(a: f32) -> f32 {
    // MLX rounds with the `+ 1.5 * 2^23` trick; an explicit round gives the
    // same integer (ties aside) and cannot be folded away by shader compilers.
    let j = f32::round(1.442695f32 * a);
    let f = fma(j, -0.693_145_75f32, a);
    let mut s = f * f;
    if a == 0.0f32 {
        s = a;
    }
    let mut r = fma(1.973_509_8e-4f32, f, 1.393_090_7e-3f32);
    r = fma(r, f, 8.333_44e-3f32);
    r = fma(r, f, 4.166_680_2e-2f32);
    r = fma(r, f, 1.666_667_2e-1f32);
    r = fma(r, f, 4.999_999_7e-1f32);
    let mut u = f;
    if j == 1.0f32 {
        u = f + 0.5f32;
    }
    let v = fma(r, s, u);
    let t = 0.5f32 * f32::powf(2.0f32, j);
    let y = t - 0.5f32;
    let x = (t - y) - 0.5f32;
    r = fma(v, t, x) + y;
    r += r;
    if j == 0.0f32 {
        r = v;
    }
    if j == 1.0f32 {
        r = v + v;
    }
    if f32::abs(a - 1.0f32) > 88.0f32 {
        let e = f32::exp(a * 0.693_147_2f32);
        r = fma(e, e, -1.0f32);
    }
    r
}

/// MLX `erf` for `|a| > 0.9277344` (`mathfns.jl`).
#[cube]
fn mlx_erf_large(a: f32) -> f32 {
    let t = f32::abs(a);
    let s = a * a;
    let mut r = fma(-1.728_534_7e-5f32, t, 3.831_971_3e-4f32);
    let u = fma(-3.883_964_4e-3f32, t, 2.425_462_2e-2f32);
    r = fma(r, s, u);
    r = fma(r, t, -1.067_778_8e-1f32);
    r = fma(r, t, -6.348_466_9e-1f32);
    r = fma(r, t, -1.287_175_1e-1f32);
    r = fma(r, t, -t);
    let e = 0.0f32 - mlx_expm1f(r);
    select(a < 0.0f32, 0.0f32 - e, e)
}

/// MLX `erf` for `|a| <= 0.9277344` (`mathfns.jl`).
#[cube]
fn mlx_erf_small(a: f32) -> f32 {
    let s = a * a;
    let mut r = fma(-5.967_617e-4f32, s, 4.991_194_2e-3f32);
    r = fma(r, s, -2.676_813_5e-2f32);
    r = fma(r, s, 1.128_199_2e-1f32);
    r = fma(r, s, -3.761_253_4e-1f32);
    r = fma(r, s, 1.283_791_7e-1f32);
    fma(r, a, a)
}

/// `erf` as evaluated by the MLX Metal kernel (`mathfns.jl`).
#[cube]
fn mlx_erf(a: f32) -> f32 {
    select(
        f32::abs(a) > 0.927_734_4f32,
        mlx_erf_large(a),
        mlx_erf_small(a),
    )
}

/// Exact (erf) GELU with MLX's erf.
#[cube]
pub fn mlx_gelu(x: f32) -> f32 {
    x * (1.0f32 + mlx_erf(x / 1.414_213_6f32)) / 2.0f32
}

/// `out[oo + i] = gelu(u[uo + r + 2I c]) * u[uo + I + r + 2I c]` for
/// `i = r + I c < total`.
#[cube(launch_unchecked)]
pub fn cube_geglu(out: &mut Array<f32>, u: &Array<f32>, oo: u32, uo: u32, inter: u32, total: u32) {
    let i = ABSOLUTE_POS_X;
    if i < total {
        let c = i / inter;
        let r = i - c * inter;
        let o = uo + 2 * inter * c;
        out[(oo + i) as usize] = mlx_gelu(u[(o + r) as usize]) * u[(o + inter + r) as usize];
    }
}

/// `y[yo + i] = act(y[yo + i] + bias[bo + i % rows])` (`act`: 0 identity,
/// 1 GELU, 2 ReLU; `has_bias` selects the bias).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn cube_bias_act(
    y: &mut Array<f32>,
    bias: &Array<f32>,
    yo: u32,
    bo: u32,
    rows: u32,
    total: u32,
    #[comptime] has_bias: bool,
    #[comptime] act: u32,
) {
    let i = ABSOLUTE_POS_X;
    if i < total {
        let mut v = y[(yo + i) as usize];
        if has_bias {
            v += bias[(bo + i % rows) as usize];
        }
        if act == 1 {
            v = mlx_gelu(v);
        }
        if act == 2 {
            v = f32::max(v, 0.0f32);
        }
        y[(yo + i) as usize] = v;
    }
}

/// `act[ao + i] = silu(gu[go + r + 2I c]) * gu[go + I + r + 2I c]`.
#[cube(launch_unchecked)]
pub fn cube_silu_mul(
    act: &mut Array<f32>,
    gu: &Array<f32>,
    ao: u32,
    go: u32,
    inter: u32,
    total: u32,
) {
    let i = ABSOLUTE_POS_X;
    if i < total {
        let c = i / inter;
        let r = i - c * inter;
        let o = go + 2 * inter * c;
        let g = gu[(o + r) as usize];
        act[(ao + i) as usize] = (g / (1.0f32 + f32::exp(-g))) * gu[(o + inter + r) as usize];
    }
}

/// DeltaNet output gate: for each (head, token) column of `output`
/// (`dv` values at `oo + dv * column`), `gated = rmsnorm(o) * w * silu(z)`
/// with `z = proj[po + z_row + head * dv + r + ld * token]`. One plane per
/// column.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn cube_rms_gate(
    gated: &mut Array<f32>,
    output: &Array<f32>,
    w: &Array<f32>,
    proj: &Array<f32>,
    go: u32,
    oo: u32,
    wo: u32,
    po: u32,
    ld: u32,
    z_row: u32,
    dv: u32,
    heads: u32,
    eps: f32,
) {
    let lane = UNIT_POS_PLANE;
    let pd = PLANE_DIM;
    let column = CUBE_POS_X;
    let head = column % heads;
    let token = column / heads;
    let base = oo + dv * column;
    let mut total = 0.0f32;
    let mut r = lane;
    while r < dv {
        let v = output[(base + r) as usize];
        total += v * v;
        r += pd;
    }
    let scale = 1.0f32 / f32::sqrt(plane_sum(total) / f32::cast_from(dv) + eps);
    let mut r = lane;
    while r < dv {
        let z = proj[(po + z_row + head * dv + r + ld * token) as usize];
        gated[(go + dv * column + r) as usize] = output[(base + r) as usize]
            * scale
            * w[(wo + r) as usize]
            * (z / (1.0f32 + f32::exp(-z)));
        r += pd;
    }
}

// ------------------------------------------------------------------ attention
//
// CubeCL port of the raw path's fused FP32 attention (`laya_raw.cu`, itself
// a port of Laya.jl's kernel) for head_dim 64: 128 units take 64 queries of
// one head, walk the keys in tiles of 32 with an online softmax, RoPE applied
// while staging Q and K, XOR-swizzled shared tiles, register-tiled 4x4 / 4x8
// outer products, and key tiles fully masked for the block (padding, outside
// the local window) skipped. CubeCL has no `__syncthreads_or`; the skip
// decisions are instead block-uniform values every unit computes from the
// token mask (a tile is skipped when none of its keys is valid; skipping is
// only an optimization, masked pairs contribute zero either way), which also
// keeps the shared memory at exactly 32 KiB (WebGPU's common limit). Shared
// loads are scalar (no `float4`).
//
// qkv is (64, H, 3, L, B) at `qo`, y is (64, H, L, B) at `yo`; cos/sin
// tables are (32, >= L) at `co`/`so` of `rope`; `valid` (L, B) at `vo` holds
// 1.0 for real tokens. `window < 0` is global attention.

/// Head width of [`cube_laya_attention`].
pub const FA_HD: u32 = 64;
/// Queries per cube of [`cube_laya_attention`].
pub const FA_BQ: u32 = 64;
/// Keys per tile of [`cube_laya_attention`].
pub const FA_BK: u32 = 32;
/// Units per cube of [`cube_laya_attention`].
pub const FA_NT: u32 = 128;

#[cube]
fn fa_qpos(q: u32, d: u32) -> u32 {
    4 * ((q >> 2) ^ (d & 15)) + (q & 3) + FA_BQ * d
}

#[cube]
fn fa_kpos(k: u32, d: u32) -> u32 {
    4 * ((k >> 2) ^ (d & 7)) + (k & 3) + FA_BK * d
}

#[cube]
fn fa_ppos(q: u32, k: u32) -> u32 {
    4 * ((q >> 2) ^ (k & 15)) + (q & 3) + FA_BQ * k
}

#[cube]
fn fa_max8(x: f32) -> f32 {
    let a = f32::max(x, plane_shuffle_xor(x, 1));
    let b = f32::max(a, plane_shuffle_xor(a, 2));
    f32::max(b, plane_shuffle_xor(b, 4))
}

#[cube]
fn fa_sum8(x: f32) -> f32 {
    let a = x + plane_shuffle_xor(x, 1);
    let b = a + plane_shuffle_xor(a, 2);
    b + plane_shuffle_xor(b, 4)
}

/// Whether any of `count` tokens from `from` (clipped to `len`) is valid;
/// the same value in every unit of the cube.
#[cube]
fn fa_any_valid(valid: &Array<f32>, vbase: u32, from: u32, count: u32, len: u32) -> bool {
    let end = u32::min(from + count, len);
    let mut any = false;
    let mut k = from;
    while k < end {
        if valid[(vbase + k) as usize] != 0.0f32 {
            any = true;
        }
        k += 1;
    }
    any
}

/// Whether any of `count` tokens from `from` (clipped to `len`) is padding.
#[cube]
fn fa_any_padding(valid: &Array<f32>, vbase: u32, from: u32, count: u32, len: u32) -> bool {
    let end = u32::min(from + count, len);
    let mut any = false;
    let mut k = from;
    while k < end {
        if valid[(vbase + k) as usize] == 0.0f32 {
            any = true;
        }
        k += 1;
    }
    any
}

/// Stage `n` rows (tokens `l0..`) of q or k (`p` = 0, 1) into a shared tile,
/// RoPE and `scale` applied; zeros past `len`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn fa_stage_rope(
    tile: &mut SharedMemory<f32>,
    qkv: &Array<f32>,
    rope: &Array<f32>,
    qo: u32,
    co: u32,
    so: u32,
    p: u32,
    l0: u32,
    n: u32,
    len: u32,
    h: u32,
    heads: u32,
    b: u32,
    scale: f32,
    #[comptime] is_q: bool,
    #[comptime] userope: bool,
) {
    let half = FA_HD / 2;
    let mut idx = UNIT_POS_X;
    while idx < n * half {
        let j = idx % half;
        let row = idx / half;
        let l = l0 + row;
        let mut x1 = 0.0f32;
        let mut x2 = 0.0f32;
        let mut c = 1.0f32;
        let mut s = 0.0f32;
        if l < len {
            let src = qo + FA_HD * (h + heads * (p + 3 * (l + len * b)));
            x1 = qkv[(src + j) as usize];
            x2 = qkv[(src + j + half) as usize];
            if userope {
                c = rope[(co + j + half * l) as usize];
                s = rope[(so + j + half * l) as usize];
            }
        }
        let mut p1 = fa_kpos(row, j);
        let mut p2 = fa_kpos(row, j + half);
        if is_q {
            p1 = fa_qpos(row, j);
            p2 = fa_qpos(row, j + half);
        }
        tile[p1 as usize] = (x1 * c - x2 * s) * scale;
        tile[p2 as usize] = (x1 * s + x2 * c) * scale;
        idx += FA_NT;
    }
}

/// Fused attention (see the section notes). grid (ceil(L / 64), H, B),
/// 128 units per cube.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn cube_laya_attention(
    y: &mut Array<f32>,
    qkv: &Array<f32>,
    rope: &Array<f32>,
    valid: &Array<f32>,
    yo: u32,
    qo: u32,
    co: u32,
    so: u32,
    vo: u32,
    window: i32,
    heads: u32,
    len: u32,
    scale: f32,
    #[comptime] userope: bool,
) {
    let mut qs = SharedMemory::<f32>::new(4096usize);
    let mut ks = SharedMemory::<f32>::new(2048usize);
    let mut vs = SharedMemory::<f32>::new(2048usize);
    let q0 = CUBE_POS_X * FA_BQ;
    let h = CUBE_POS_Y;
    let b = CUBE_POS_Z;
    let t = UNIT_POS_X;
    let lane = t % 32;
    let kg = lane % 8;
    let qg = lane / 8 + 4 * (t / 32);
    let vbase = vo + len * b;

    fa_stage_rope(
        &mut qs, qkv, rope, qo, co, so, 0, q0, FA_BQ, len, h, heads, b, scale, true, userope,
    );

    let mut win = Array::<u32>::new(4usize);
    #[unroll]
    for i in 0usize..4 {
        let qi = q0 + 4 * qg + i as u32;
        let mut w = 0u32;
        if window >= 0 && qi < len {
            if valid[(vbase + qi) as usize] != 0.0f32 {
                w = 1;
            }
        }
        win[i] = w;
    }
    let ntiles = (len + FA_BK - 1) / FA_BK;
    let lastq = u32::min(q0 + FA_BQ, len);
    // A padded query keeps every valid key, so the window cannot narrow the
    // key range of a block containing one.
    let padded = fa_any_padding(valid, vbase, q0, FA_BQ, len);
    let mut t0 = 0u32;
    let mut t1 = ntiles;
    if window >= 0 && !padded {
        let w = u32::cast_from(window);
        let lo = select(q0 > w, q0 - w, 0u32);
        t0 = lo / FA_BK;
        t1 = u32::min((lastq - 1 + w) / FA_BK + 1, ntiles);
    }
    let mut m = Array::<f32>::new(4usize);
    let mut l = Array::<f32>::new(4usize);
    let mut acc = Array::<f32>::new(32usize);
    #[unroll]
    for i in 0usize..4 {
        m[i] = f32::new(-1.0e30f32);
        l[i] = 0.0f32;
    }
    #[unroll]
    for e in 0usize..32 {
        acc[e] = 0.0f32;
    }
    let mut tile = t0;
    while tile < t1 {
        let k0 = tile * FA_BK;
        let mut bits = 0u32;
        #[unroll]
        for j in 0usize..4 {
            let kk = k0 + 4 * kg + j as u32;
            let mut kv = false;
            if kk < len {
                kv = valid[(vbase + kk) as usize] != 0.0f32;
            }
            #[unroll]
            for i in 0usize..4 {
                let qi = q0 + 4 * qg + i as u32;
                let diff = i32::cast_from(qi) - i32::cast_from(kk);
                let inside = win[i] == 0 || (diff <= window && -diff <= window);
                if qi < len && kv && inside {
                    bits |= 1u32 << (j + 4usize * i) as u32;
                }
            }
        }
        if fa_any_valid(valid, vbase, k0, FA_BK, len) {
            sync_cube(); // the previous tile's reads of ks / vs are done
            fa_stage_rope(
                &mut ks, qkv, rope, qo, co, so, 1, k0, FA_BK, len, h, heads, b, 1.0f32, false,
                userope,
            );
            let mut idx = t;
            while idx < FA_HD * FA_BK {
                let i = idx % FA_HD;
                let kl = k0 + idx / FA_HD;
                let mut v = 0.0f32;
                if kl < len {
                    v = qkv[(qo + FA_HD * (h + heads * (2 + 3 * (kl + len * b))) + i) as usize];
                }
                vs[idx as usize] = v;
                idx += FA_NT;
            }
            sync_cube();
            let mut s = Array::<f32>::new(16usize);
            #[unroll]
            for e in 0usize..16 {
                s[e] = 0.0f32;
            }
            let mut qa = Array::<f32>::new(4usize);
            let mut ka = Array::<f32>::new(4usize);
            for d in 0..FA_HD {
                let qp = fa_qpos(4 * qg, d);
                let kp = fa_kpos(4 * kg, d);
                #[unroll]
                for c in 0usize..4 {
                    qa[c] = qs[(qp + c as u32) as usize];
                    ka[c] = ks[(kp + c as u32) as usize];
                }
                #[unroll]
                for e in 0usize..16 {
                    s[e] = fma(qa[e / 4usize], ka[e % 4usize], s[e]);
                }
            }
            let mut p = Array::<f32>::new(16usize);
            let mut alpha = Array::<f32>::new(4usize);
            #[unroll]
            for i in 0usize..4 {
                let mut mx = f32::new(-1.0e30f32);
                #[unroll]
                for j in 0usize..4 {
                    if (bits >> (j + 4usize * i) as u32) & 1 == 1 {
                        mx = f32::max(mx, s[4usize * i + j]);
                    }
                }
                let mn = f32::max(m[i], fa_max8(mx));
                alpha[i] = f32::exp(m[i] - mn);
                let mut sum = 0.0f32;
                #[unroll]
                for j in 0usize..4 {
                    let mut pv = 0.0f32;
                    if (bits >> (j + 4usize * i) as u32) & 1 == 1 {
                        pv = f32::exp(s[4usize * i + j] - mn);
                    }
                    p[4usize * i + j] = pv;
                    sum += pv;
                }
                l[i] = l[i] * alpha[i] + fa_sum8(sum);
                m[i] = mn;
            }
            #[unroll]
            for e in 0usize..32 {
                acc[e] = acc[e] * alpha[e / 8usize];
            }
            sync_cube();
            #[unroll]
            for j in 0usize..4 {
                let pp = fa_ppos(4 * qg, 4 * kg + j as u32);
                #[unroll]
                for i in 0usize..4 {
                    ks[(pp + i as u32) as usize] = p[j + 4usize * i];
                }
            }
            sync_cube();
            let mut pa = Array::<f32>::new(4usize);
            let mut va = Array::<f32>::new(8usize);
            for k in 0..FA_BK {
                let pp = fa_ppos(4 * qg, k);
                #[unroll]
                for i in 0usize..4 {
                    pa[i] = ks[(pp + i as u32) as usize];
                }
                #[unroll]
                for r in 0usize..8 {
                    va[r] = vs[(8 * kg + r as u32 + FA_HD * k) as usize];
                }
                #[unroll]
                for e in 0usize..32 {
                    acc[e] = fma(va[e % 8usize], pa[e / 8usize], acc[e]);
                }
            }
        }
        tile += 1;
    }
    #[unroll]
    for i in 0usize..4 {
        let qi = q0 + 4 * qg + i as u32;
        if qi < len {
            let mut inv = 0.0f32;
            if l[i] > 0.0f32 {
                inv = 1.0f32 / l[i];
            }
            let o = yo + 8 * kg + FA_HD * (h + heads * (qi + len * b));
            #[unroll]
            for r in 0usize..8 {
                y[(o + r as u32) as usize] = acc[r + 8usize * i] * inv;
            }
        }
    }
}

// ---------------------------------------------------------- chunked delta

/// Tokens per chunk of [`cube_chunk_prepare`] (the raw path's `GD_CHUNK`).
pub const GD_CHUNK: u32 = 64;
/// Token ranges (cube rows) per chunk of [`cube_chunk_prepare`].
pub const GD_PARTS: u32 = 4;

/// CubeCL port of `gd_chunk_prepare` (`delta_raw.cu`): the operands of the
/// chunked WY delta rule for batch `b = head + heads * chunk`. Every buffer
/// lives in the one scratch array `s` at the given element offsets:
/// `kq` (dk, 2C, B) = [k | q], `scaled` (dk + dv, C, B) = [beta exp(G) k;
/// beta v], `dq` / `ek` (dk, C, B) = exp(G) q / exp(G_C - G) k, `cum` /
/// `gates` (C, B) = G / beta, `growth` (B) = exp(G_C). Tokens past the
/// sequence are zero with zero decay. grid (B, GD_PARTS), 128 units.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn cube_chunk_prepare(
    s: &mut Array<f32>,
    kq: u32,
    scaled: u32,
    dq: u32,
    ek: u32,
    cum: u32,
    gates: u32,
    growth: u32,
    query: u32,
    key: u32,
    mixed: u32,
    ld_mixed: u32,
    beta: u32,
    decay: u32,
    dk: u32,
    dv: u32,
    value_start: u32,
    heads: u32,
    groups: u32,
    tokens: u32,
) {
    let mut logs = SharedMemory::<f32>::new(64usize);
    let mut betas = SharedMemory::<f32>::new(64usize);
    let th = UNIT_POS_X;
    let threads = CUBE_DIM_X;
    let batch = CUBE_POS_X;
    let part = CUBE_POS_Y;
    let head = batch % heads;
    let first = (batch / heads) * GD_CHUNK;
    let key_heads = heads / groups;
    let key_head = head / groups;
    let width = dk + dv;
    if th < GD_CHUNK {
        let token = first + th;
        let mut lg = 0.0f32;
        let mut bt = 0.0f32;
        if token < tokens {
            lg = s[(decay + head + heads * token) as usize];
            bt = s[(beta + head + heads * token) as usize];
        }
        logs[th as usize] = lg;
        betas[th as usize] = bt;
    }
    let mut offset = 1u32;
    while offset < GD_CHUNK {
        sync_cube();
        let mut value = 0.0f32;
        if th < GD_CHUNK && th >= offset {
            value = logs[(th - offset) as usize];
        }
        sync_cube();
        if th < GD_CHUNK && th >= offset {
            logs[th as usize] += value;
        }
        offset *= 2;
    }
    sync_cube();
    let last = logs[(GD_CHUNK - 1) as usize];
    if part == 0 && th < GD_CHUNK {
        s[(cum + th + GD_CHUNK * batch) as usize] = logs[th as usize];
        s[(gates + th + GD_CHUNK * batch) as usize] = betas[th as usize];
        if th == GD_CHUNK - 1 {
            s[(growth + batch) as usize] = f32::exp(last);
        }
    }
    let span = GD_CHUNK / GD_PARTS;
    let mut t = part * span;
    while t < (part + 1) * span {
        let token = first + t;
        let valid = token < tokens;
        let g = f32::exp(logs[t as usize]);
        let ending = f32::exp(last - logs[t as usize]);
        let bt = betas[t as usize];
        let ko = dk * (key_head + key_heads * token);
        let mut c = th;
        while c < dk {
            let mut k = 0.0f32;
            let mut q = 0.0f32;
            if valid {
                k = s[(key + ko + c) as usize];
                q = s[(query + ko + c) as usize];
            }
            s[(kq + c + dk * (t + 2 * GD_CHUNK * batch)) as usize] = k;
            s[(kq + c + dk * (GD_CHUNK + t + 2 * GD_CHUNK * batch)) as usize] = q;
            s[(dq + c + dk * (t + GD_CHUNK * batch)) as usize] = q * g;
            s[(ek + c + dk * (t + GD_CHUNK * batch)) as usize] = k * ending;
            s[(scaled + c + width * (t + GD_CHUNK * batch)) as usize] = k * bt * g;
            c += threads;
        }
        let mut c = th;
        while c < dv {
            let mut v = 0.0f32;
            if valid {
                v = s[(mixed + value_start + head * dv + c + ld_mixed * token) as usize];
            }
            s[(scaled + dk + c + width * (t + GD_CHUNK * batch)) as usize] = v * bt;
            c += threads;
        }
        t += 1;
    }
}
