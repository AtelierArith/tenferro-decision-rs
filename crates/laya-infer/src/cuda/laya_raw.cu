// Raw CUDA kernels of Laya's single-stream device forward (`cuda_raw.rs`).
//
// Activations are feature-major `(d, T)` with token column `t = l + L * b`
// (the host reference layout). The Rust driver validates every shape, owns
// all buffers, and enqueues these kernels and the cuBLAS products on one
// stream. Pointers arrive as 64-bit scalars.

#define LAYA_WARP 32
#define LAYA_COLS 8
#define LAYA_NEG_INF (-__int_as_float(0x7f800000))

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// `expm1` / `erf` as evaluated by the MLX Metal kernel (`mathfns.jl`); the
// constants are MLX's bits.
__device__ float mlx_expm1f(float a) {
    float j = fmaf(1.442695f, a, 12582912.0f);
    j = j - 12582912.0f;
    int i = (int)j;
    float f = fmaf(j, -0.693145752f, a);
    float s = a == 0.0f ? a : f * f;
    float r = 1.9735098e-4f;
    r = fmaf(r, f, 1.3930907e-3f);
    r = fmaf(r, f, 8.33344e-3f);
    r = fmaf(r, f, 4.1666802e-2f);
    r = fmaf(r, f, 1.6666672e-1f);
    r = fmaf(r, f, 4.9999997e-1f);
    float u = j == 1.0f ? f + 0.5f : f;
    float v = fmaf(r, s, u);
    float t = 0.5f * ldexpf(1.0f, i);
    float y = t - 0.5f;
    float x = (t - y) - 0.5f;
    r = fmaf(v, t, x) + y;
    r = r + r;
    if (j == 0.0f) r = v;
    if (j == 1.0f) r = v + v;
    if (fabsf(a - 1.0f) > 88.0f) {
        float e = exp2f(a);
        r = fmaf(e, e, -1.0f);
    }
    return r;
}

__device__ float mlx_erf(float a) {
    float t = fabsf(a);
    float s = a * a;
    if (t > 0.9277344f) {
        float r = fmaf(-1.7285347e-5f, t, 3.8319713e-4f);
        float u = fmaf(-3.8839644e-3f, t, 2.4254622e-2f);
        r = fmaf(r, s, u);
        r = fmaf(r, t, -1.0677788e-1f);
        r = fmaf(r, t, -6.3484669e-1f);
        r = fmaf(r, t, -1.2871751e-1f);
        r = fmaf(r, t, -t);
        r = -mlx_expm1f(r);
        return copysignf(r, a);
    }
    float r = -5.967617e-4f;
    r = fmaf(r, s, 4.9911942e-3f);
    r = fmaf(r, s, -2.6768135e-2f);
    r = fmaf(r, s, 1.1281992e-1f);
    r = fmaf(r, s, -3.7612534e-1f);
    r = fmaf(r, s, 1.2837917e-1f);
    return fmaf(r, a, a);
}

__device__ __forceinline__ float laya_gelu(float x) {
    return x * (1.0f + mlx_erf(x / 1.41421356237309515f)) / 2.0f;
}

// out (d, T) = columns ids[t] of table (d, vocab).
extern "C" __global__ void laya_embed(float* out, const float* table, const int* ids, int d, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int t = i / d;
    int r = i - t * d;
    out[i] = table[r + (long long)d * ids[t]];
}

// LayerNorm over each column of x (d, ncol) into y (may alias x); one warp
// per column, LAYA_COLS columns per block. Each lane keeps its d / 32 values
// in registers (d <= 32 * LAYA_NV). bias may be null.
#define LAYA_NV 32
extern "C" __global__ void laya_layernorm(
    float* y, const float* x, const float* w, const float* bias, int d, int ncol, float eps) {
    int lane = threadIdx.x & 31;
    int col = blockIdx.x * LAYA_COLS + (threadIdx.x >> 5);
    if (col >= ncol) return;
    const float* src = x + (long long)col * d;
    float v[LAYA_NV];
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < LAYA_NV; ++k) {
        int i = lane + 32 * k;
        v[k] = i < d ? src[i] : 0.0f;
        s += v[k];
    }
    float mu = warp_sum(s) / (float)d;
    float q = 0.0f;
#pragma unroll
    for (int k = 0; k < LAYA_NV; ++k) {
        int i = lane + 32 * k;
        if (i < d) {
            float c = v[k] - mu;
            q += c * c;
        }
    }
    float inv = 1.0f / sqrtf(warp_sum(q) / (float)d + eps);
    float* dst = y + (long long)col * d;
#pragma unroll
    for (int k = 0; k < LAYA_NV; ++k) {
        int i = lane + 32 * k;
        if (i < d) {
            float t = (v[k] - mu) * inv * w[i];
            if (bias != 0) t += bias[i];
            dst[i] = t;
        }
    }
}

// out (I, ncol) = gelu(u[0:I]) * u[I:2I] for u (2I, ncol).
extern "C" __global__ void laya_geglu(float* out, const float* u, int inter, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = i / inter;
    int r = i - c * inter;
    long long o = 2LL * inter * c;
    out[i] = laya_gelu(u[o + r]) * u[o + inter + r];
}

// y (rows, ncol) = act(y + bias[row]) in place; bias may be null.
// act 0: identity, 1: GELU, 2: ReLU.
extern "C" __global__ void laya_bias_act(float* y, const float* bias, int rows, int total, int act) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float v = y[i];
    if (bias != 0) v += bias[i % rows];
    if (act == 1) v = laya_gelu(v);
    else if (act == 2) v = fmaxf(v, 0.0f);
    y[i] = v;
}

// x (d, L, B) += type_emb[:, qtype[b]].
extern "C" __global__ void laya_add_type(float* x, const float* type_emb, const int* qtype, int d, int length, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int col = i / d;
    int r = i - col * d;
    int b = col / length;
    x[i] += type_emb[r + d * qtype[b]];
}

// markers (d, K, B) = h[:, max(pos, 0), b].
extern "C" __global__ void laya_gather_markers(
    float* out, const float* h, const int* pos, int d, int k_count, int length, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int col = i / d;
    int r = i - col * d;
    int b = col / k_count;
    int p = pos[col];
    p = p < 0 ? 0 : p;
    out[i] = h[r + (long long)d * (p + (long long)length * b)];
}

// One block (32 threads) per batch row: mask the raw marker logits (K, B),
// write them to `masked`, and build pooled (d + 4, B) = [h[:, 0, b];
// top1; top1 - top2; entropy; used / 255] (`pool_host`).
extern "C" __global__ void laya_pool(
    float* masked, float* pooled, const float* raw, const int* marker_mask, const float* h,
    int k_count, int d, int length) {
    int b = blockIdx.x;
    int lane = threadIdx.x;
    for (int i = lane; i < d; i += 32) pooled[i + (long long)(d + 4) * b] = h[i + (long long)d * length * b];
    if (lane != 0) return;
    float mx = LAYA_NEG_INF;
    int count = 0;
    for (int j = 0; j < k_count; ++j) {
        int idx = j + k_count * b;
        float v = marker_mask[idx] ? raw[idx] : -1.0e4f;
        masked[idx] = v;
        mx = fmaxf(mx, v);
        count += marker_mask[idx] ? 1 : 0;
    }
    float sum = 0.0f;
    for (int j = 0; j < k_count; ++j) sum += expf(masked[j + k_count * b] - mx);
    float top1 = LAYA_NEG_INF, top2 = LAYA_NEG_INF, entropy = 0.0f;
    for (int j = 0; j < k_count; ++j) {
        float p = expf(masked[j + k_count * b] - mx) / sum;
        if (p > top1) {
            top2 = top1;
            top1 = p;
        } else if (p > top2) {
            top2 = p;
        }
        entropy -= p * logf(fmaxf(p, 1.0e-9f));
    }
    float used = (float)(count > 2 ? count : 2);
    entropy /= logf(used);
    float* f = pooled + (long long)(d + 4) * b + d;
    f[0] = top1;
    f[1] = top1 - top2;
    f[2] = entropy;
    f[3] = used / 255.0f;
}

// ------------------------------------------------------------------ attention
//
// Port of Laya.jl's fused FP32 attention (`LayaCUDAExt.jl`) for head_dim 64:
// a block of 128 threads takes 64 queries of one head and walks the keys in
// tiles of 32 with an online softmax. Q (RoPE and scale applied) stays in
// shared memory; each K tile is staged with RoPE, S = K'Q and O += V P run as
// register-tiled outer products (4 queries x 4 keys, 4 queries x 8 rows per
// thread, 128-bit shared loads on XOR-swizzled tiles). Key tiles fully masked
// for the block (padding, outside the local window) are skipped. Nothing but
// the output touches device memory.
//
// qkv is (64, H, 3, L, B) (the fused projection), y is (64, H, L, B);
// cos/sin tables are (32, >= L); valid is (L, B) (nonzero = real token);
// window < 0 is global attention. A valid query keeps the valid keys within
// `window`; a padded query keeps every valid key.

#define FA_HD 64
#define FA_BQ 64
#define FA_BK 32
#define FA_NT 128

__device__ __forceinline__ int qpos(int q, int d) { return 4 * ((q >> 2) ^ (d & 15)) + (q & 3) + FA_BQ * d; }
__device__ __forceinline__ int kpos(int k, int d) { return 4 * ((k >> 2) ^ (d & 7)) + (k & 3) + FA_BK * d; }
__device__ __forceinline__ int ppos(int q, int k) { return 4 * ((q >> 2) ^ (k & 15)) + (q & 3) + FA_BQ * k; }

__device__ __forceinline__ long long fa_offset(int p, int l, int h, int H, int L, int b) {
    return (long long)FA_HD * (h + (long long)H * (p + 3LL * (l + (long long)L * b)));
}

template <bool Q>
__device__ __forceinline__ void stage_rope(
    float* tile, const float* qkv, const float* c, const float* s, int p, int l0, int n, int L,
    int h, int H, int b, int userope, float scale, int t) {
    const int half = FA_HD / 2;
    for (int idx = t; idx < n * half; idx += FA_NT) {
        int j = idx % half;
        int row = idx / half;
        int l = l0 + row;
        float x1 = 0.0f, x2 = 0.0f;
        if (l < L) {
            long long src = fa_offset(p, l, h, H, L, b);
            x1 = qkv[src + j];
            x2 = qkv[src + j + half];
        }
        int p1 = Q ? qpos(row, j) : kpos(row, j);
        int p2 = Q ? qpos(row, j + half) : kpos(row, j + half);
        if (userope && l < L) {
            float cc = c[j + half * l];
            float ss = s[j + half * l];
            tile[p1] = (x1 * cc - x2 * ss) * scale;
            tile[p2] = (x1 * ss + x2 * cc) * scale;
        } else {
            tile[p1] = x1 * scale;
            tile[p2] = x2 * scale;
        }
    }
}

__device__ __forceinline__ float fa_max(float x) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 1));
    x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 2));
    return fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 4));
}

__device__ __forceinline__ float fa_sum(float x) {
    x += __shfl_xor_sync(0xffffffffu, x, 1);
    x += __shfl_xor_sync(0xffffffffu, x, 2);
    return x + __shfl_xor_sync(0xffffffffu, x, 4);
}

extern "C" __global__ void __launch_bounds__(FA_NT) laya_attention(
    float* y, const float* qkv, const float* c, const float* s, const int* valid,
    int window, int H, int L, int userope, float scale) {
    __shared__ __align__(16) float Qs[FA_BQ * FA_HD];
    __shared__ __align__(16) float Ks[FA_BK * FA_HD];  // K tile, then P
    __shared__ __align__(16) float Vs[FA_HD * FA_BK];
    const int q0 = blockIdx.x * FA_BQ;
    const int h = blockIdx.y, b = blockIdx.z;
    const int t = threadIdx.x;
    const int lane = t & 31;
    const int kg = lane & 7;                   // keys 4kg..4kg+3 of S, rows 8kg..8kg+7 of O
    const int qg = (lane >> 3) + 4 * (t >> 5); // queries 4qg..4qg+3
    const int vo = L * b;

    stage_rope<true>(Qs, qkv, c, s, 0, q0, FA_BQ, L, h, H, b, userope, scale, t);

    int qis[4];
    bool win[4];
    bool any_pad = false;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        qis[i] = q0 + 4 * qg + i;
        win[i] = window >= 0 && qis[i] < L && valid[vo + qis[i]] != 0;
        any_pad |= qis[i] < L && !win[i];
    }
    const int ntiles = (L + FA_BK - 1) / FA_BK;
    const int lastq = min(q0 + FA_BQ, L);
    const int padded = __syncthreads_or(any_pad);
    int t0, t1;
    if (window < 0 || padded) {
        t0 = 0;
        t1 = ntiles - 1;
    } else {
        t0 = max(q0 - window, 0) / FA_BK;
        t1 = min((lastq - 1 + window) / FA_BK, ntiles - 1);
    }
    float m[4], l[4], acc[32];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        m[i] = LAYA_NEG_INF;
        l[i] = 0.0f;
    }
#pragma unroll
    for (int e = 0; e < 32; ++e) acc[e] = 0.0f;

    for (int tile = t0; tile <= t1; ++tile) {
        const int k0 = tile * FA_BK;
        int kks[4];
        bool kv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            kks[j] = k0 + 4 * kg + j;
            kv[j] = kks[j] < L && valid[vo + kks[j]] != 0;
        }
        int bits = 0;  // (key j, query i) kept at bit j + 4 i
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                bool keep = qis[i] < L && kv[j] && (!win[i] || abs(qis[i] - kks[j]) <= window);
                bits |= (keep ? 1 : 0) << (j + 4 * i);
            }
        if (!__syncthreads_or(bits != 0)) continue;  // also orders the previous tile's reads
        stage_rope<false>(Ks, qkv, c, s, 1, k0, FA_BK, L, h, H, b, userope, 1.0f, t);
        for (int idx = t; idx < FA_HD * FA_BK; idx += FA_NT) {
            int i = idx % FA_HD;
            int kl = k0 + idx / FA_HD;
            Vs[idx] = kl < L ? qkv[fa_offset(2, kl, h, H, L, b) + i] : 0.0f;
        }
        __syncthreads();
        float S[16];
#pragma unroll
        for (int e = 0; e < 16; ++e) S[e] = 0.0f;
#pragma unroll 8
        for (int d = 0; d < FA_HD; ++d) {
            float4 qv = *reinterpret_cast<const float4*>(Qs + qpos(4 * qg, d));
            float4 k4 = *reinterpret_cast<const float4*>(Ks + kpos(4 * kg, d));
            float qa[4] = {qv.x, qv.y, qv.z, qv.w};
            float ka[4] = {k4.x, k4.y, k4.z, k4.w};
#pragma unroll
            for (int e = 0; e < 16; ++e) S[e] = fmaf(qa[e >> 2], ka[e & 3], S[e]);
        }
        float P[16];
        float mn[4], alpha[4];
#pragma unroll
        for (int e = 0; e < 16; ++e) S[e] = ((bits >> e) & 1) ? S[e] : LAYA_NEG_INF;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float mx = fmaxf(fmaxf(S[4 * i], S[4 * i + 1]), fmaxf(S[4 * i + 2], S[4 * i + 3]));
            mn[i] = fmaxf(m[i], fa_max(mx));
            alpha[i] = mn[i] == m[i] ? 1.0f : expf(m[i] - mn[i]);
        }
#pragma unroll
        for (int e = 0; e < 16; ++e) P[e] = S[e] == LAYA_NEG_INF ? 0.0f : expf(S[e] - mn[e >> 2]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            l[i] = l[i] * alpha[i] + fa_sum(P[4 * i] + P[4 * i + 1] + P[4 * i + 2] + P[4 * i + 3]);
            m[i] = mn[i];
        }
#pragma unroll
        for (int e = 0; e < 32; ++e) acc[e] *= alpha[e >> 3];
        __syncthreads();  // every thread is done with the K tile
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            *reinterpret_cast<float4*>(Ks + ppos(4 * qg, 4 * kg + j)) =
                make_float4(P[j], P[j + 4], P[j + 8], P[j + 12]);
        }
        __syncthreads();
#pragma unroll 4
        for (int k = 0; k < FA_BK; ++k) {
            float4 pv = *reinterpret_cast<const float4*>(Ks + ppos(4 * qg, k));
            float4 v1 = *reinterpret_cast<const float4*>(Vs + 8 * kg + FA_HD * k);
            float4 v2 = *reinterpret_cast<const float4*>(Vs + 8 * kg + 4 + FA_HD * k);
            float pa[4] = {pv.x, pv.y, pv.z, pv.w};
            float va[8] = {v1.x, v1.y, v1.z, v1.w, v2.x, v2.y, v2.z, v2.w};
#pragma unroll
            for (int e = 0; e < 32; ++e) acc[e] = fmaf(va[e & 7], pa[e >> 3], acc[e]);
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (qis[i] < L) {
            float inv_l = l[i] > 0.0f ? 1.0f / l[i] : 0.0f;
            long long o = 8 * kg + (long long)FA_HD * (h + (long long)H * (qis[i] + (long long)L * b));
#pragma unroll
            for (int r = 0; r < 8; ++r) y[o + r] = acc[r + 8 * i] * inv_l;
        }
    }
}
