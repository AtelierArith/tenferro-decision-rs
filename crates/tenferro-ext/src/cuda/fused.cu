// Fused f32 CUDA kernels for the decision engines' device forward.
//
// Every tensor is a dense column-major buffer; the Rust adapter
// (`cuda_fused.rs`) validates shapes, layouts, residency and launch geometry
// and owns all operands until its scope's completion fence. Each kernel
// replaces a chain of small eager operations whose launch overhead dominates
// small-batch inference.

#define FUSED_MAX_THREADS 1024
#define FUSED_NEG_INF (-__int_as_float(0x7f800000))

__device__ __forceinline__ float fused_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ float fused_warp_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

// Block-wide sum for blockDim.x a multiple of 32 (<= 1024). `scratch` holds at
// least 32 floats. Every thread returns the total.
__device__ float fused_block_sum(float value, float* scratch) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = (blockDim.x + 31) >> 5;
    value = fused_warp_sum(value);
    __syncthreads();
    if (lane == 0) scratch[warp] = value;
    __syncthreads();
    float total = lane < warps ? scratch[lane] : 0.0f;
    if (warp == 0) {
        total = fused_warp_sum(total);
        if (lane == 0) scratch[0] = total;
    }
    __syncthreads();
    return scratch[0];
}

__device__ float fused_block_max(float value, float* scratch) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = (blockDim.x + 31) >> 5;
    value = fused_warp_max(value);
    __syncthreads();
    if (lane == 0) scratch[warp] = value;
    __syncthreads();
    float total = lane < warps ? scratch[lane] : FUSED_NEG_INF;
    if (warp == 0) {
        total = fused_warp_max(total);
        if (lane == 0) scratch[0] = total;
    }
    __syncthreads();
    return scratch[0];
}

__device__ __forceinline__ float fused_gelu_erf(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
}

__device__ __forceinline__ float fused_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Normalize `vectors` vectors of `width` features. Element (v, f) of x and
// out lives at v * vec_stride + f * feat_stride. One block per vector.
// mode 0: RMSNorm x * w; 1: centered RMSNorm x * (1 + w);
// 2: LayerNorm (x - mean) * w (+ bias when non-null).
extern "C" __global__ void fused_norm(
    float* out, const float* x, const float* weight, const float* bias,
    int vectors, int width, int vec_stride, int feat_stride, float eps, int mode) {
    __shared__ float scratch[32];
    int vector = blockIdx.x;
    if (vector >= vectors) return;
    const float* row = x + (long long)vector * vec_stride;
    float* dst = out + (long long)vector * vec_stride;
    float mean = 0.0f;
    if (mode == 2) {
        float sum = 0.0f;
        for (int f = threadIdx.x; f < width; f += blockDim.x) sum += row[(long long)f * feat_stride];
        mean = fused_block_sum(sum, scratch) / (float)width;
    }
    float squares = 0.0f;
    for (int f = threadIdx.x; f < width; f += blockDim.x) {
        float value = row[(long long)f * feat_stride] - mean;
        squares += value * value;
    }
    float inv = 1.0f / sqrtf(fused_block_sum(squares, scratch) / (float)width + eps);
    for (int f = threadIdx.x; f < width; f += blockDim.x) {
        float value = (row[(long long)f * feat_stride] - mean) * inv;
        float scale = mode == 1 ? 1.0f + weight[f] : weight[f];
        float result = value * scale;
        if (mode == 2 && bias != 0) result += bias[f];
        dst[(long long)f * feat_stride] = result;
    }
}

// gu is (rows, 2 * inter) with column blocks gate | up; out (rows, inter) =
// silu(gate) * up.
extern "C" __global__ void fused_gated_silu_split(
    float* out, const float* gu, int rows, int inter) {
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long n = (long long)rows * inter;
    if (index >= n) return;
    float g = gu[index];
    out[index] = (g / (1.0f + expf(-g))) * gu[n + index];
}

// out = silu(gate) * up, elementwise over n values.
extern "C" __global__ void fused_gated_silu(
    float* out, const float* gate, const float* up, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;
    float g = gate[index];
    out[index] = (g / (1.0f + expf(-g))) * up[index];
}

// u is (2 * inter, cols); out (inter, cols) = gelu_erf(u[:inter]) * u[inter:].
extern "C" __global__ void fused_geglu_cols(
    float* out, const float* u, int inter, int cols) {
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (long long)inter * cols) return;
    long long col = index / inter;
    long long row = index % inter;
    float value = u[row + col * 2 * inter];
    float gate = u[inter + row + col * 2 * inter];
    out[index] = fused_gelu_erf(value) * gate;
}

// x is (rows, cols); out = act(x + bias[row]) where bias may be null.
// act 0: identity, 1: exact (erf) GELU, 2: ReLU.
extern "C" __global__ void fused_bias_act(
    float* out, const float* x, const float* bias, int rows, int cols, int act) {
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (long long)rows * cols) return;
    float value = x[index];
    if (bias != 0) value += bias[index % rows];
    if (act == 1) value = fused_gelu_erf(value);
    else if (act == 2) value = fmaxf(value, 0.0f);
    out[index] = value;
}

// RoPE cos/sin tables are (length, rotary_dim / 2) row-major, computed on the
// host in f64 (as the native composition does) and rounded to f32.
// Jeff full attention preparation. qkvg is the stacked time-first projection
// (length, 4 * width) with column blocks q | k | v | gate, width = heads *
// head_dim; grid (length, heads, 3), block head_dim threads rounded up to a
// multiple of 32. z = 0/1: centered per-head RMSNorm with (1 + norm) then
// partial RoPE over the first rotary_dim channels (pairs i, i + rotary_dim /
// 2) into qh/kh; z = 2: copy v into vh. Outputs are head-major
// (heads, length, head_dim).
extern "C" __global__ void fused_jeff_qkv_prep(
    float* qh, float* kh, float* vh, const float* qkvg,
    const float* q_norm, const float* k_norm, const float* cos_table, const float* sin_table,
    int length, int heads, int head_dim, int rotary_dim, float eps) {
    __shared__ float scratch[32];
    extern __shared__ float values[];
    int token = blockIdx.x;
    int head = blockIdx.y;
    int which = blockIdx.z;
    int d = threadIdx.x;
    int width = heads * head_dim;
    bool active = d < head_dim;
    float value = active
        ? qkvg[token + (long long)(which * width + head * head_dim + d) * length] : 0.0f;
    long long dst_index = ((long long)head * length + token) * head_dim + d;
    if (which == 2) {
        if (active) vh[dst_index] = value;
        return;
    }
    float squares = fused_block_sum(value * value, scratch);
    float inv = 1.0f / sqrtf(squares / (float)head_dim + eps);
    const float* norm = which == 0 ? q_norm : k_norm;
    float normed = active ? value * inv * (1.0f + norm[d]) : 0.0f;
    if (active) values[d] = normed;
    __syncthreads();
    if (!active) return;
    float result = normed;
    int half = rotary_dim / 2;
    if (d < rotary_dim) {
        int i = d < half ? d : d - half;
        float c = cos_table[(long long)token * half + i];
        float s = sin_table[(long long)token * half + i];
        result = d < half ? values[d] * c - values[d + half] * s
                          : values[d - half] * s + values[d] * c;
    }
    (which == 0 ? qh : kh)[dst_index] = result;
}

// Laya attention preparation. qkv is feature-first (3 * hidden, length,
// batch); grid (length, heads, 3 * batch), block head_dim threads. q/k get
// full-width rotate-half RoPE when base > 0; v is copied. Outputs are
// head-major (batch, heads, length, head_dim).
extern "C" __global__ void fused_laya_qkv_prep(
    float* qh, float* kh, float* vh, const float* qkv,
    const float* cos_table, const float* sin_table,
    int hidden, int heads, int length, int batch, int rotate) {
    extern __shared__ float values[];
    int token = blockIdx.x;
    int head = blockIdx.y;
    int which = blockIdx.z % 3;
    int b = blockIdx.z / 3;
    int head_dim = hidden / heads;
    int d = threadIdx.x;
    bool active = d < head_dim;
    long long src = (long long)which * hidden + head * head_dim + d
        + 3LL * hidden * (token + (long long)length * b);
    float value = active ? qkv[src] : 0.0f;
    long long dst_index = (((long long)b * heads + head) * length + token) * head_dim + d;
    float* dst = which == 0 ? qh : (which == 1 ? kh : vh);
    if (which == 2 || !rotate) {
        if (active) dst[dst_index] = value;
        return;
    }
    if (active) values[d] = value;
    __syncthreads();
    if (!active) return;
    int half = head_dim / 2;
    int i = d < half ? d : d - half;
    float c = cos_table[(long long)token * half + i];
    float s = sin_table[(long long)token * half + i];
    dst[dst_index] = d < half ? values[d] * c - values[d + half] * s
                              : values[d - half] * s + values[d] * c;
}

// Softmax attention over head-major q/k/v (batch, heads, length, head_dim).
// grid (length, heads, batch); block a multiple of 32 threads; dynamic shared
// memory (length + head_dim) floats. Keys are dropped when `causal` and
// key > query, or when `active` (batch, length) is non-null and zero; a
// non-null `bias` (length_q, length_k, batch) is added to the scaled scores.
// A row without any kept key writes zeros. The output (and optional sigmoid
// `gate`, same layout) element (token, feature, b) lives at
// token * o_tok + feature * o_feat + b * o_batch with feature = head * head_dim + d.
extern "C" __global__ void fused_attention(
    float* out, const float* qh, const float* kh, const float* vh,
    const float* bias, const float* active, const float* gate,
    int length, int heads, int head_dim, int batch, float scale, int causal,
    long long o_tok, long long o_feat, long long o_batch, long long gate_offset) {
    __shared__ float scratch[32];
    extern __shared__ float shared[];
    float* scores = shared;
    float* query = shared + length;
    int token = blockIdx.x;
    int head = blockIdx.y;
    int b = blockIdx.z;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    long long base = ((long long)b * heads + head) * length * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        query[d] = qh[base + (long long)token * head_dim + d];
    }
    __syncthreads();
    for (int key = warp; key < length; key += warps) {
        bool keep = (!causal || key <= token)
            && (active == 0 || active[(long long)b * length + key] != 0.0f);
        float dot = 0.0f;
        if (keep) {
            const float* kv = kh + base + (long long)key * head_dim;
            for (int d = lane; d < head_dim; d += 32) dot += query[d] * kv[d];
            dot = fused_warp_sum(dot);
        }
        if (lane == 0) {
            float score = FUSED_NEG_INF;
            if (keep) {
                score = dot * scale;
                if (bias != 0) score += bias[token + (long long)length * (key + (long long)length * b)];
            }
            scores[key] = score;
        }
    }
    __syncthreads();
    float local_max = FUSED_NEG_INF;
    for (int key = threadIdx.x; key < length; key += blockDim.x) local_max = fmaxf(local_max, scores[key]);
    float max = fused_block_max(local_max, scratch);
    float local_sum = 0.0f;
    for (int key = threadIdx.x; key < length; key += blockDim.x) {
        float score = scores[key];
        float p = score == FUSED_NEG_INF ? 0.0f : expf(score - max);
        scores[key] = p;
        local_sum += p;
    }
    float sum = fused_block_sum(local_sum, scratch);
    float inverse = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        const float* vcol = vh + base + d;
        for (int key = 0; key < length; ++key) {
            float p = scores[key];
            if (p != 0.0f) acc += p * vcol[(long long)key * head_dim];
        }
        acc *= inverse;
        long long index = token * o_tok + (long long)(head * head_dim + d) * o_feat + b * o_batch;
        if (gate != 0) acc *= fused_sigmoid(gate[gate_offset + index]);
        out[index] = acc;
    }
}

// Token embedding gather for ids given as exact f32 values. Element
// (token, feature) of out lives at token * o_tok + feature * o_feat; the
// table element (feature, id) at feature * t_feat + id * t_tok. Ids are
// validated on the host.
extern "C" __global__ void fused_embedding_rows(
    float* out, const float* table, const float* ids, int length, int hidden,
    long long o_tok, long long o_feat, long long t_feat, long long t_tok) {
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (long long)length * hidden) return;
    long long token = index % length;
    long long feature = index / length;
    long long id = (long long)ids[token];
    out[token * o_tok + feature * o_feat] = table[feature * t_feat + id * t_tok];
}
