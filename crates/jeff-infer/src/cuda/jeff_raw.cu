// Raw CUDA kernels of Jeff's single-stream device forward (`cuda_raw.rs`).
//
// Activations are feature-major (rows, T): element (row, t) of a buffer with
// leading dimension `ld` lives at row + ld * t. The Rust driver validates
// every shape and owns all buffers; pointers arrive as 64-bit scalars.

#define JEFF_COLS 8
#define JEFF_NV 32
#define JEFF_NEG_INF (-__int_as_float(0x7f800000))

__device__ __forceinline__ float jeff_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float jeff_warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Block-wide sum (blockDim a multiple of 32); `scratch` holds 32 floats.
__device__ float jeff_block_sum(float v, float* scratch) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    v = jeff_warp_sum(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < warps; ++w) total += scratch[w];
    return total;
}

// out (H, T) = embedding columns: table is (H, vocab) row-major (d * vocab + id).
extern "C" __global__ void jeff_embed(float* out, const float* table, const int* ids, int hidden, long long vocab, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int t = i / hidden;
    int d = i - t * hidden;
    out[i] = table[(long long)d * vocab + ids[t]];
}

// y = x * rsqrt(mean(x^2) + eps) * (1 + w) per column (centered RMSNorm),
// times mask[col] when mask is non-null. One warp per column, JEFF_COLS
// columns per block; columns have length `hidden` (<= 32 * JEFF_NV) and
// stride `hidden` in both x and y.
extern "C" __global__ void jeff_rms(
    float* y, const float* x, const float* w, const float* mask, int hidden, int ncol, float eps) {
    int lane = threadIdx.x & 31;
    int col = blockIdx.x * JEFF_COLS + (threadIdx.x >> 5);
    if (col >= ncol) return;
    const float* src = x + (long long)col * hidden;
    float v[JEFF_NV];
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < JEFF_NV; ++k) {
        int i = lane + 32 * k;
        v[k] = i < hidden ? src[i] : 0.0f;
        s += v[k] * v[k];
    }
    float scale = 1.0f / sqrtf(jeff_warp_sum(s) / (float)hidden + eps);
    if (mask != 0) scale *= mask[col];
    float* dst = y + (long long)col * hidden;
#pragma unroll
    for (int k = 0; k < JEFF_NV; ++k) {
        int i = lane + 32 * k;
        if (i < hidden) dst[i] = v[k] * scale * (1.0f + w[i]);
    }
}

// act (I, T) = silu(gu[0:I]) * gu[I:2I] for gu (2I, T).
extern "C" __global__ void jeff_silu_mul(float* act, const float* gu, int inter, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = i / inter;
    int r = i - c * inter;
    long long o = 2LL * inter * c;
    float g = gu[o + r];
    act[i] = (g / (1.0f + expf(-g))) * gu[o + inter + r];
}

// Per-head centered RMSNorm (scale 1 + norm) and partial RoPE over the first
// `rotary` channels (pairs i, i + rotary / 2) of rows `row0 + head * hd` of
// proj (ld, T), for tokens col0 .. col0 + gridDim.x - 1. out is
// (hd, gridDim.x, heads). cos/sin tables are (rotary / 2, positions); the
// RoPE position of token t is t. block = hd threads (a multiple of 32).
extern "C" __global__ void jeff_attn_prep(
    float* out, const float* proj, int ld, int row0, const float* norm, const float* cos_table,
    const float* sin_table, int hd, int rotary, int col0, float eps) {
    __shared__ float scratch[32];
    extern __shared__ float values[];
    int i = blockIdx.x;
    int head = blockIdx.y;
    int n = gridDim.x;
    int token = col0 + i;
    int d = threadIdx.x;
    float value = d < hd ? proj[row0 + head * hd + d + (long long)ld * token] : 0.0f;
    float squares = jeff_block_sum(value * value, scratch);
    float inv = 1.0f / sqrtf(squares / (float)hd + eps);
    bool active = d < hd;
    float normed = active ? value * inv * (1.0f + norm[d]) : 0.0f;
    if (active) values[d] = normed;
    __syncthreads();
    if (!active) return;
    float result = normed;
    int half = rotary / 2;
    if (d < rotary) {
        int j = d < half ? d : d - half;
        float c = cos_table[j + (long long)half * token];
        float s = sin_table[j + (long long)half * token];
        result = d < half ? values[d] * c - values[d + half] * s : values[d - half] * s + values[d] * c;
    }
    out[d + (long long)hd * (i + (long long)n * head)] = result;
}

// Block-wide max (blockDim a multiple of 32); `scratch` holds 32 floats.
__device__ float jeff_block_max(float v, float* scratch) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    v = jeff_warp_max(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float total = JEFF_NEG_INF;
    for (int w = 0; w < warps; ++w) total = fmaxf(total, scratch[w]);
    __syncthreads();
    return total;
}

// Gated causal softmax attention for queries col0 .. col0 + gridDim.x - 1.
// qh is (hd, gridDim.x, heads) and kh (hd, T, kv_heads) from
// `jeff_attn_prep`; v is rows v_row + kv_head * hd and the query gate rows
// g_row + head * hd of proj (ld, T). Query q keeps keys k <= q with
// mask[k] != 0; a query without kept keys writes zeros. out is
// (hd * heads, gridDim.x) = sigmoid(gate) * softmax(q k' * scale) v.
// grid (n, heads); block a multiple of 32 threads; dynamic shared memory
// (hd + T) floats.
extern "C" __global__ void jeff_attention(
    float* out, const float* qh, const float* kh, const float* proj, int ld, int v_row, int g_row,
    const float* mask, int hd, int heads, int kv_heads, int tokens, int col0, float scale) {
    __shared__ float scratch[32];
    extern __shared__ float shared[];
    float* query = shared;
    float* scores = shared + hd;
    int i = blockIdx.x;
    int head = blockIdx.y;
    int n = gridDim.x;
    int qi = col0 + i;
    int kv_head = head / (heads / kv_heads);
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) query[d] = qh[d + (long long)hd * (i + (long long)n * head)];
    __syncthreads();
    int keys = qi + 1;
    for (int key = warp; key < keys; key += warps) {
        bool keep = mask[key] != 0.0f;
        float dot = 0.0f;
        if (keep) {
            const float* k = kh + (long long)hd * (key + (long long)tokens * kv_head);
            for (int d = lane; d < hd; d += 32) dot += query[d] * k[d];
            dot = jeff_warp_sum(dot);
        }
        if (lane == 0) scores[key] = keep ? dot * scale : JEFF_NEG_INF;
    }
    __syncthreads();
    float local = JEFF_NEG_INF;
    for (int key = threadIdx.x; key < keys; key += blockDim.x) local = fmaxf(local, scores[key]);
    float mx = jeff_block_max(local, scratch);
    float sum = 0.0f;
    for (int key = threadIdx.x; key < keys; key += blockDim.x) {
        float s = scores[key];
        float p = s == JEFF_NEG_INF ? 0.0f : expf(s - mx);
        scores[key] = p;
        sum += p;
    }
    float total = jeff_block_sum(sum, scratch);
    __syncthreads();
    float inverse = total > 0.0f ? 1.0f / total : 0.0f;
    const float* v = proj + v_row + kv_head * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.0f;
        for (int key = 0; key < keys; ++key) {
            float p = scores[key];
            if (p != 0.0f) acc += p * v[d + (long long)ld * key];
        }
        float gate = proj[g_row + head * hd + d + (long long)ld * qi];
        out[d + (long long)hd * head + (long long)hd * heads * i] = acc * inverse / (1.0f + expf(-gate));
    }
}

// `jeff_attention` for JEFF_QT consecutive queries per block, so every K and
// V row read from memory serves JEFF_QT queries. grid (ceil(n / JEFF_QT),
// heads) with n the query count; block 256 threads (>= JEFF_QT warps);
// dynamic shared memory JEFF_QT * (hd + T) floats.
#define JEFF_QT 8
extern "C" __global__ void jeff_attention_tile(
    float* out, const float* qh, const float* kh, const float* proj, int ld, int v_row, int g_row,
    const float* mask, int hd, int heads, int kv_heads, int tokens, int col0, int n, float scale) {
    __shared__ float inv[JEFF_QT];
    extern __shared__ float shared[];
    float* qs = shared;                  // (hd, JEFF_QT)
    float* ps = shared + JEFF_QT * hd;   // (T, JEFF_QT) scores, then probabilities
    int i0 = blockIdx.x * JEFF_QT;
    int head = blockIdx.y;
    int nt = min(JEFF_QT, n - i0);
    int kv_head = head / (heads / kv_heads);
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    for (int e = threadIdx.x; e < JEFF_QT * hd; e += blockDim.x) {
        int j = e / hd;
        int d = e - j * hd;
        qs[e] = j < nt ? qh[d + (long long)hd * (i0 + j + (long long)n * head)] : 0.0f;
    }
    __syncthreads();
    int q_first = col0 + i0;
    int keys = q_first + nt;
    for (int key = warp; key < keys; key += warps) {
        bool keep = mask[key] != 0.0f;
        float dots[JEFF_QT];
#pragma unroll
        for (int j = 0; j < JEFF_QT; ++j) dots[j] = 0.0f;
        if (keep) {
            const float* k = kh + (long long)hd * (key + (long long)tokens * kv_head);
            for (int d = lane; d < hd; d += 32) {
                float kv = k[d];
#pragma unroll
                for (int j = 0; j < JEFF_QT; ++j) dots[j] += qs[j * hd + d] * kv;
            }
#pragma unroll
            for (int j = 0; j < JEFF_QT; ++j) dots[j] = jeff_warp_sum(dots[j]);
        }
        if (lane < JEFF_QT) {
            int j = lane;
            float dot = dots[0];
#pragma unroll
            for (int jj = 1; jj < JEFF_QT; ++jj) dot = jj == j ? dots[jj] : dot;
            bool kept = keep && j < nt && key <= q_first + j;
            ps[j * tokens + key] = kept ? dot * scale : JEFF_NEG_INF;
        }
    }
    __syncthreads();
    if (warp < JEFF_QT) {
        float* row = ps + warp * tokens;
        float mx = JEFF_NEG_INF;
        for (int key = lane; key < keys; key += 32) mx = fmaxf(mx, row[key]);
        mx = jeff_warp_max(mx);
        float sum = 0.0f;
        for (int key = lane; key < keys; key += 32) {
            float s = row[key];
            float p = s == JEFF_NEG_INF ? 0.0f : expf(s - mx);
            row[key] = p;
            sum += p;
        }
        sum = jeff_warp_sum(sum);
        if (lane == 0) inv[warp] = sum > 0.0f ? 1.0f / sum : 0.0f;
    }
    __syncthreads();
    const float* v = proj + v_row + kv_head * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc[JEFF_QT];
#pragma unroll
        for (int j = 0; j < JEFF_QT; ++j) acc[j] = 0.0f;
        for (int key = 0; key < keys; ++key) {
            float vv = v[d + (long long)ld * key];
#pragma unroll
            for (int j = 0; j < JEFF_QT; ++j) acc[j] += ps[j * tokens + key] * vv;
        }
#pragma unroll
        for (int j = 0; j < JEFF_QT; ++j) {
            if (j < nt) {
                int qi = q_first + j;
                float gate = proj[g_row + head * hd + d + (long long)ld * qi];
                out[d + (long long)hd * head + (long long)hd * heads * (i0 + j)] =
                    acc[j] * inv[j] / (1.0f + expf(-gate));
            }
        }
    }
}
