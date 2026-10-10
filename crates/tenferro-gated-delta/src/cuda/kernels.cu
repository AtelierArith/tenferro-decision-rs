// Low-level f32 stages for a device-resident Gated DeltaNet pipeline.
// Buffers use (length, channels) column-major storage: channel * length + token.
// Launch contracts are intentionally explicit; the Rust adapter must validate
// shapes, residency, aliasing and resource lifetimes before enqueueing.

extern "C" __global__ void gated_delta_conv_silu(
    float* output, const float* input, const float* weight,
    int length, int channels, int taps) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length * channels) return;
    int channel = index / length;
    int token = index % length;
    float value = 0.0f;
    for (int tap = 0; tap < taps; ++tap) {
        int lag = taps - 1 - tap;
        if (token >= lag) {
            value += input[channel * length + token - lag] * weight[tap * channels + channel];
        }
    }
    output[index] = value / (1.0f + expf(-value));
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

// One warp per (value head, value row); a block holds `blockDim.x / 32`
// consecutive rows of one value head: block=(32 * rows,1,1),
// grid=(value_heads, ceil(value_dim / rows),1). key_dim must be in 1..=256.
// Each block stages `tile` tokens of Q/K in dynamic shared memory
// (2 * tile * key_dim floats, coalesced along tokens), L2-normalizes them
// once per block, then every warp advances its register-resident state row
// through the tile. Beta/decay gates are fused; output RMSNorm needs the
// separate epilogue because its reduction spans independently owned rows.
extern "C" __global__ void gated_delta_recurrent(
    float* output, const float* mixed, const float* a, const float* b,
    const float* a_decay, const float* dt_bias,
    int length, int key_dim, int value_dim, int key_heads, int value_heads, int tile) {
    extern __shared__ float staged[];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    // Layout: q[tile][key_dim], k[tile][key_dim], factor[tile], beta[tile],
    // v[tile][warps]. The host sizes the allocation accordingly.
    float* qs = staged;
    float* ks = qs + tile * key_dim;
    float* factors = ks + tile * key_dim;
    float* betas = factors + tile;
    float* vs = betas + tile;
    int head = blockIdx.x;
    int row0 = blockIdx.y * warps;
    int row = row0 + warp;
    bool owns_row = row < value_dim;
    int key_head = head / (value_heads / key_heads);
    int key_width = key_heads * key_dim;
    int q_offset = key_head * key_dim;
    int k_offset = key_width + q_offset;
    int v_base = 2 * key_width + head * value_dim;
    float state[8] = {};
    float scale = 1.0f / sqrtf((float)key_dim);
    float decay = a_decay[head];
    float bias = dt_bias[head];
    for (int start = 0; start < length; start += tile) {
        int count = min(tile, length - start);
        __syncthreads();
        // Stage Q/K (tokens fastest, coalesced), gates and this block's V.
        for (int i = threadIdx.x; i < count * key_dim; i += blockDim.x) {
            int d = i / count;
            int t = i % count;
            qs[t * key_dim + d] = mixed[(long long)(q_offset + d) * length + start + t];
            ks[t * key_dim + d] = mixed[(long long)(k_offset + d) * length + start + t];
        }
        for (int t = threadIdx.x; t < count; t += blockDim.x) {
            int token = start + t;
            float raw_a = a[head * length + token] + bias;
            float softplus = fmaxf(raw_a, 0.0f) + log1pf(expf(-fabsf(raw_a)));
            factors[t] = expf(decay * softplus);
            betas[t] = 1.0f / (1.0f + expf(-b[head * length + token]));
        }
        for (int i = threadIdx.x; i < count * warps; i += blockDim.x) {
            int r = i / count;
            int t = i % count;
            vs[t * warps + r] = row0 + r < value_dim
                ? mixed[(long long)(v_base + row0 + r) * length + start + t] : 0.0f;
        }
        __syncthreads();
        // Normalize each staged token once per block (one warp per token).
        for (int t = warp; t < count; t += warps) {
            float qn = 0.0f, kn = 0.0f;
            for (int d = lane; d < key_dim; d += 32) {
                float qv = qs[t * key_dim + d];
                float kv = ks[t * key_dim + d];
                qn += qv * qv;
                kn += kv * kv;
            }
            float q_inv = 1.0f / sqrtf(warp_sum(qn) + 1e-6f);
            float k_inv = 1.0f / sqrtf(warp_sum(kn) + 1e-6f);
            for (int d = lane; d < key_dim; d += 32) {
                qs[t * key_dim + d] = (qs[t * key_dim + d] * q_inv) * scale;
                ks[t * key_dim + d] *= k_inv;
            }
        }
        __syncthreads();
        if (!owns_row) continue;
        for (int t = 0; t < count; ++t) {
            float q[8], k[8];
            float prediction = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int d = lane + 32 * j;
                q[j] = d < key_dim ? qs[t * key_dim + d] : 0.0f;
                k[j] = d < key_dim ? ks[t * key_dim + d] : 0.0f;
                prediction += state[j] * k[j];
            }
            prediction = warp_sum(prediction);
            float factor = factors[t];
            float correction = betas[t] * (vs[t * warps + warp] - factor * prediction);
            float result = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                state[j] = fmaf(correction, k[j], factor * state[j]);
                result += state[j] * q[j];
            }
            result = warp_sum(result);
            if (lane == 0) output[(long long)(head * value_dim + row) * length + start + t] = result;
        }
    }
}

// One block per (value head, token), blockDim.x a power of two in 32..=1024.
// Dynamic shared memory: blockDim.x * sizeof(float). Input/output do not alias.
extern "C" __global__ void gated_delta_norm_gate(
    float* output, const float* input, const float* z, const float* norm,
    int length, int value_dim, float eps) {
    int head = blockIdx.x / length;
    int token = blockIdx.x % length;
    int lane = threadIdx.x;
    extern __shared__ float scratch[];
    float square_sum = 0.0f;
    for (int row = lane; row < value_dim; row += blockDim.x) {
        float value = input[(head * value_dim + row) * length + token];
        square_sum += value * value;
    }
    scratch[lane] = square_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (lane < offset) scratch[lane] += scratch[lane + offset];
        __syncthreads();
    }
    float inv = 1.0f / sqrtf(scratch[0] / (float)value_dim + eps);
    for (int row = lane; row < value_dim; row += blockDim.x) {
        int index = (head * value_dim + row) * length + token;
        float gate = z[index];
        output[index] = (input[index] * inv * norm[row]) * (gate / (1.0f + expf(-gate)));
    }
}

// One block per (value head, chunk), block=(128,1,1), grid=(heads*chunks,1,1).
// chunk_size is in 1..=256. Prefix/beta/tail use (length,heads) column-major;
// pair uses (chunk_size,chunk_size,chunks,heads) column-major, including zeros
// for upper triangle and padded rows/columns. final uses (chunks,heads).
// Store cumulative LOG decay so a final factor that underflows to zero does
// not destroy the finite differences used by pair/tail weights.
extern "C" __global__ void gated_delta_chunk_decay(
    float* cumulative, float* beta, float* pair, float* tail, float* final_factor,
    const float* a, const float* b, const float* a_decay, const float* dt_bias,
    int length, int heads, int chunk_size) {
    int chunks = (length - 1) / chunk_size + 1;
    int task = blockIdx.x;
    if (task >= heads * chunks) return;
    int head = task / chunks;
    int chunk = task % chunks;
    int start = chunk * chunk_size;
    int count = min(chunk_size, length - start);
    int base = head * length + start;
    for (int row = threadIdx.x; row < count; row += blockDim.x) {
        float sum = 0.0f;
        for (int t = 0; t <= row; ++t) {
            float raw_a = a[base + t] + dt_bias[head];
            float softplus = fmaxf(raw_a, 0.0f) + log1pf(expf(-fabsf(raw_a)));
            sum += a_decay[head] * softplus;
        }
        cumulative[base + row] = sum;
        beta[base + row] = 1.0f / (1.0f + expf(-b[base + row]));
    }
    __syncthreads();
    float last = cumulative[base + count - 1];
    if (threadIdx.x == 0) final_factor[task] = expf(last);
    for (int row = threadIdx.x; row < count; row += blockDim.x) {
        tail[base + row] = expf(last - cumulative[base + row]);
    }
    int cells = chunk_size * chunk_size;
    for (int cell = threadIdx.x; cell < cells; cell += blockDim.x) {
        int row = cell % chunk_size;
        int column = cell / chunk_size;
        pair[task * cells + cell] = row < count && column < count && row >= column
            ? expf(cumulative[base + row] - cumulative[base + column]) : 0.0f;
    }
}
