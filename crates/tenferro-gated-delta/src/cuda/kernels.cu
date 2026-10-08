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

// Exactly one warp per (value head, value row): block=(32,1,1),
// grid=(value_heads,value_dim,1). key_dim must be in 1..=256.
// State remains in per-lane registers throughout the token scan. Q/K
// normalization and beta/decay gates are fused; output RMSNorm needs the
// separate epilogue because its reduction spans independently owned rows.
extern "C" __global__ void gated_delta_recurrent(
    float* output, const float* mixed, const float* a, const float* b,
    const float* a_decay, const float* dt_bias,
    int length, int key_dim, int value_dim, int key_heads, int value_heads) {
    int lane = threadIdx.x;
    int head = blockIdx.x;
    int row = blockIdx.y;
    int key_head = head / (value_heads / key_heads);
    int key_width = key_heads * key_dim;
    int q_offset = key_head * key_dim;
    int k_offset = key_width + q_offset;
    int v_offset = 2 * key_width + head * value_dim + row;
    float state[8] = {};
    float scale = 1.0f / sqrtf((float)key_dim);
    for (int token = 0; token < length; ++token) {
        float q[8], k[8];
        float q_norm = 0.0f, k_norm = 0.0f;
        #pragma unroll
        for (int tile = 0; tile < 8; ++tile) {
            int d = lane + 32 * tile;
            q[tile] = d < key_dim ? mixed[(q_offset + d) * length + token] : 0.0f;
            k[tile] = d < key_dim ? mixed[(k_offset + d) * length + token] : 0.0f;
            q_norm += q[tile] * q[tile];
            k_norm += k[tile] * k[tile];
        }
        float q_inv = 1.0f / sqrtf(warp_sum(q_norm) + 1e-6f);
        float k_inv = 1.0f / sqrtf(warp_sum(k_norm) + 1e-6f);
        float prediction = 0.0f;
        #pragma unroll
        for (int tile = 0; tile < 8; ++tile) {
            q[tile] = (q[tile] * q_inv) * scale;
            k[tile] *= k_inv;
            prediction += state[tile] * k[tile];
        }
        prediction = warp_sum(prediction);
        float raw_a = a[head * length + token] + dt_bias[head];
        float softplus = fmaxf(raw_a, 0.0f) + log1pf(expf(-fabsf(raw_a)));
        float factor = expf(a_decay[head] * softplus);
        float beta = 1.0f / (1.0f + expf(-b[head * length + token]));
        float correction = beta * (mixed[v_offset * length + token] - factor * prediction);
        float result = 0.0f;
        #pragma unroll
        for (int tile = 0; tile < 8; ++tile) {
            state[tile] = fmaf(correction, k[tile], factor * state[tile]);
            result += state[tile] * q[tile];
        }
        result = warp_sum(result);
        if (lane == 0) output[(head * value_dim + row) * length + token] = result;
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
