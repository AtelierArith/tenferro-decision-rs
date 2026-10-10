// Raw Gated DeltaNet kernels for the single-stream device forward
// (`raw.rs`). Activations are feature-major (rows, tokens): element
// (row, t) of a buffer with leading dimension `ld` lives at row + ld * t.
// The chunked scan is the WY form of QwenDecisionCore.jl's
// `cuda_delta_chunked.jl` (flash-linear-attention's algorithm): within a
// chunk of GD_CHUNK tokens the recurrence becomes small batched products
// (cuBLAS, issued by the Rust driver) and only the state hand-off between
// chunks is sequential. Batch index b = head + heads * chunk throughout.

#define GD_CHUNK 64
#define GD_BLOCK 16
#define GD_PARTS 4
#define GD_CONV_TOKENS 8

__device__ __forceinline__ float gd_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float gd_silu(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ float gd_softplus(float x) {
    return fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
}

// out (channels, T) = silu(causal depthwise conv of input rows 0..channels);
// input has leading dimension ld_in, weight is (taps, channels) row-major.
// Thread = one channel over GD_CONV_TOKENS consecutive tokens (taps <= 8).
extern "C" __global__ void gd_conv_silu(
    float* out, const float* input, const float* weight, int channels, int tokens, int ld_in, int taps) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    int first = blockIdx.y * GD_CONV_TOKENS;
    if (ch >= channels) return;
    float w[8];
    float window[8];
    for (int tap = 0; tap < taps; ++tap) w[tap] = weight[tap * channels + ch];
    for (int lag = 0; lag < taps - 1; ++lag) {
        int src = first - (taps - 1) + lag;
        window[lag] = src >= 0 ? input[ch + (long long)ld_in * src] : 0.0f;
    }
    for (int t = 0; t < GD_CONV_TOKENS; ++t) {
        int token = first + t;
        if (token >= tokens) break;
        window[taps - 1] = input[ch + (long long)ld_in * token];
        float value = 0.0f;
        for (int tap = 0; tap < taps; ++tap) value += window[tap] * w[tap];
        out[ch + (long long)channels * token] = gd_silu(value);
        for (int lag = 0; lag < taps - 1; ++lag) window[lag] = window[lag + 1];
    }
}

// query/key (dk, key_heads, T) from the conv output mixed (ld_mixed, T):
// q rows h*dk + c, k rows key_heads*dk + h*dk + c. One warp per (head,
// token); q is L2-normalized and scaled by 1/sqrt(dk), k L2-normalized.
extern "C" __global__ void gd_qk_normalize(
    float* query, float* key, const float* mixed, int dk, int key_heads, int ld_mixed) {
    int lane = threadIdx.x;
    int group = blockIdx.x;
    int head = group % key_heads;
    int token = group / key_heads;
    int width = dk * key_heads;
    const float* col = mixed + (long long)ld_mixed * token;
    float qs[8], ks[8];
    float qsum = 0.0f, ksum = 0.0f;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
        int c = lane + 32 * p;
        qs[p] = c < dk ? col[head * dk + c] : 0.0f;
        ks[p] = c < dk ? col[width + head * dk + c] : 0.0f;
        qsum += qs[p] * qs[p];
        ksum += ks[p] * ks[p];
    }
    float qscale = 1.0f / (sqrtf(gd_warp_sum(qsum) + 1.0e-6f) * sqrtf((float)dk));
    float kscale = 1.0f / sqrtf(gd_warp_sum(ksum) + 1.0e-6f);
    long long o = (long long)dk * (head + (long long)key_heads * token);
#pragma unroll
    for (int p = 0; p < 8; ++p) {
        int c = lane + 32 * p;
        if (c < dk) {
            query[o + c] = qs[p] * qscale;
            key[o + c] = ks[p] * kscale;
        }
    }
}

// beta = sigmoid(b), decay = a_decay * softplus(a + dt_bias), both (heads,
// T), from rows a_row.. and b_row.. of proj (ld_proj, T).
extern "C" __global__ void gd_gates(
    float* beta, float* decay, const float* proj, int ld_proj, int a_row, int b_row,
    const float* a_decay, const float* dt_bias, int heads, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int h = i % heads;
    int t = i / heads;
    float b = proj[b_row + h + (long long)ld_proj * t];
    float a = proj[a_row + h + (long long)ld_proj * t];
    beta[i] = 1.0f / (1.0f + expf(-b));
    decay[i] = a_decay[h] * gd_softplus(a + dt_bias[h]);
}

// Recurrent scan: each warp owns one value row of one head, each lane
// PARTS key components of the state. block (32 * 4), grid (dv / 4, heads).
// output is (dv, heads, T); v is row value_start + h*dv + row of mixed.
template <int PARTS>
__device__ void gd_recurrent_impl(
    float* output, const float* query, const float* key, const float* mixed, int ld_mixed,
    const float* beta, const float* decay, int dk, int dv, int value_start, int heads,
    int groups, int tokens) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int head = blockIdx.y;
    int key_heads = heads / groups;
    int key_head = head / groups;
    float state[PARTS];
#pragma unroll
    for (int p = 0; p < PARTS; ++p) state[p] = 0.0f;
    bool owns = row < dv;
    for (int t = 0; t < tokens; ++t) {
        long long ko = (long long)dk * (key_head + (long long)key_heads * t);
        float ks[PARTS], qs[PARTS];
#pragma unroll
        for (int p = 0; p < PARTS; ++p) {
            int c = lane + 32 * p;
            ks[p] = c < dk ? key[ko + c] : 0.0f;
            qs[p] = c < dk ? query[ko + c] : 0.0f;
        }
        float factor = expf(decay[head + heads * t]);
        float pred = 0.0f;
#pragma unroll
        for (int p = 0; p < PARTS; ++p) {
            state[p] *= factor;
            pred += state[p] * ks[p];
        }
        pred = gd_warp_sum(pred);
        float v = owns ? mixed[value_start + head * dv + row + (long long)ld_mixed * t] : 0.0f;
        float corr = (v - pred) * beta[head + heads * t];
        float res = 0.0f;
#pragma unroll
        for (int p = 0; p < PARTS; ++p) {
            state[p] += corr * ks[p];
            res += state[p] * qs[p];
        }
        res = gd_warp_sum(res);
        if (lane == 0 && owns) output[row + (long long)dv * (head + (long long)heads * t)] = res;
    }
}

extern "C" __global__ void gd_recurrent(
    float* output, const float* query, const float* key, const float* mixed, int ld_mixed,
    const float* beta, const float* decay, int dk, int dv, int value_start, int heads,
    int groups, int tokens) {
    if (dk <= 128) {
        gd_recurrent_impl<4>(output, query, key, mixed, ld_mixed, beta, decay, dk, dv, value_start, heads, groups, tokens);
    } else {
        gd_recurrent_impl<8>(output, query, key, mixed, ld_mixed, beta, decay, dk, dv, value_start, heads, groups, tokens);
    }
}

// Inclusive prefix sum of a chunk's log decays into shared `logs` (blockDim
// >= GD_CHUNK); `betas` receives the gates. Tokens past the sequence get 0.
__device__ void gd_chunk_logs(
    float* logs, float* betas, const float* beta, const float* decay, int head, int heads,
    int first, int tokens) {
    int th = threadIdx.x;
    if (th < GD_CHUNK) {
        int token = first + th;
        bool valid = token < tokens;
        logs[th] = valid ? decay[head + heads * token] : 0.0f;
        betas[th] = valid ? beta[head + heads * token] : 0.0f;
    }
    for (int offset = 1; offset < GD_CHUNK; offset *= 2) {
        __syncthreads();
        float value = 0.0f;
        if (th < GD_CHUNK && th >= offset) value = logs[th - offset];
        __syncthreads();
        if (th < GD_CHUNK && th >= offset) logs[th] += value;
    }
    __syncthreads();
}

// grid (batches, GD_PARTS), block 128. Writes the chunk operands:
//   kq[:, 0:C] = k, kq[:, C:2C] = q                       (dk, 2C, batches)
//   scaled[0:dk] = beta exp(G) k, scaled[dk:] = beta v    (dk+dv, C, batches)
//   dq = exp(G) q, ek = exp(G_C - G) k                    (dk, C, batches)
//   cumulative = G, gates = beta (C, batches), growth = exp(G_C) (batches)
extern "C" __global__ void gd_chunk_prepare(
    float* kq, float* scaled, float* dq, float* ek, float* cumulative, float* gates, float* growth,
    const float* query, const float* key, const float* mixed, int ld_mixed, const float* beta,
    const float* decay, int dk, int dv, int value_start, int heads, int groups, int tokens) {
    const int C = GD_CHUNK;
    __shared__ float logs[GD_CHUNK];
    __shared__ float betas[GD_CHUNK];
    int th = threadIdx.x;
    int threads = blockDim.x;
    int batch = blockIdx.x;
    int part = blockIdx.y;
    int head = batch % heads;
    int first = (batch / heads) * C;
    int key_heads = heads / groups;
    int key_head = head / groups;
    int width = dk + dv;
    gd_chunk_logs(logs, betas, beta, decay, head, heads, first, tokens);
    float last_log = logs[C - 1];
    if (part == 0 && th < C) {
        cumulative[th + C * batch] = logs[th];
        gates[th + C * batch] = betas[th];
        if (th == C - 1) growth[batch] = expf(last_log);
    }
    int span = C / GD_PARTS;
    for (int t = part * span; t < (part + 1) * span; ++t) {
        int token = first + t;
        bool valid = token < tokens;
        float g = expf(logs[t]);
        float ending = expf(last_log - logs[t]);
        long long ko = (long long)dk * (key_head + (long long)key_heads * token);
        for (int c = th; c < dk; c += threads) {
            float k = valid ? key[ko + c] : 0.0f;
            float q = valid ? query[ko + c] : 0.0f;
            kq[c + (long long)dk * (t + 2LL * C * batch)] = k;
            kq[c + (long long)dk * (C + t + 2LL * C * batch)] = q;
            dq[c + (long long)dk * (t + (long long)C * batch)] = q * g;
            ek[c + (long long)dk * (t + (long long)C * batch)] = k * ending;
            scaled[c + (long long)width * (t + (long long)C * batch)] = k * betas[t] * g;
        }
        for (int c = th; c < dv; c += threads) {
            float v = valid ? mixed[value_start + head * dv + c + (long long)ld_mixed * token] : 0.0f;
            scaled[dk + c + (long long)width * (t + (long long)C * batch)] = v * betas[t];
        }
    }
}

// One block of 4C threads per batch. From products = [k'k  k'q] (C, 2C):
//   inverse = (I + L)^-1, L[t, s] = beta_t exp(G_t - G_s) k_t'k_s (s < t)
//   intra[s, t] = exp(G_t - G_s) k_s'q_t (s <= t), zero otherwise.
// The inverse is blocked: diagonal GD_BLOCK blocks by forward substitution,
// then each block row as T_ij = -T_ii sum_k L_ik T_kj.
#define GD_LD (GD_CHUNK + 1)
extern "C" __global__ void gd_chunk_inverse(
    float* inverse, float* intra, const float* products, const float* cumulative, const float* gates) {
    const int C = GD_CHUNK;
    const int S = GD_BLOCK;
    __shared__ float system[GD_LD * GD_CHUNK];
    __shared__ float solved[GD_LD * GD_CHUNK];
    __shared__ float partial[(GD_BLOCK + 1) * GD_CHUNK];
    __shared__ float logs[GD_CHUNK];
    __shared__ float betas[GD_CHUNK];
    int th = threadIdx.x;
    int threads = blockDim.x;
    int batch = blockIdx.x;
    const float* prod = products + 2LL * C * C * batch;
    if (th < C) {
        logs[th] = cumulative[th + C * batch];
        betas[th] = gates[th + C * batch];
    }
    __syncthreads();
    for (int e = th; e < C * C; e += threads) {
        int i = e % C;
        int s = e / C;
        system[i + GD_LD * s] = s < i ? betas[i] * expf(logs[i] - logs[s]) * prod[i + C * s] : 0.0f;
        intra[i + C * (s + (long long)C * batch)] = s >= i ? expf(logs[s] - logs[i]) * prod[i + C * (C + s)] : 0.0f;
        solved[i + GD_LD * s] = i == s ? 1.0f : 0.0f;
    }
    __syncthreads();
    if (th < C) {
        int j = th;
        int stop = (j / S + 1) * S;
        for (int t = j + 1; t < stop; ++t) {
            float total = 0.0f;
            for (int s = j; s < t; ++s) total += system[t + GD_LD * s] * solved[s + GD_LD * j];
            solved[t + GD_LD * j] = -total;
        }
    }
    for (int row_block = 1; row_block < C / S; ++row_block) {
        int start = row_block * S;  // rows start..start+S-1, columns 0..start-1
        __syncthreads();
        for (int e = th; e < S * start; e += threads) {
            int r = e % S;
            int j = e / S;
            float total = 0.0f;
            for (int s = j; s < start; ++s) total += system[start + r + GD_LD * s] * solved[s + GD_LD * j];
            partial[r + (GD_BLOCK + 1) * j] = total;
        }
        __syncthreads();
        for (int e = th; e < S * start; e += threads) {
            int r = e % S;
            int j = e / S;
            float total = 0.0f;
            for (int u = 0; u <= r; ++u) total += solved[start + r + GD_LD * (start + u)] * partial[u + (GD_BLOCK + 1) * j];
            solved[start + r + GD_LD * j] = -total;
        }
    }
    __syncthreads();
    for (int e = th; e < C * C; e += threads) {
        int i = e % C;
        int s = e / C;
        inverse[i + C * (s + (long long)C * batch)] = solved[i + GD_LD * s];
    }
}

// state (dv * dk, heads) *= growth[base + head].
extern "C" __global__ void gd_state_scale(float* state, const float* growth, int base, int per_head, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    state[i] *= growth[base + i / per_head];
}

// gated (dv * heads, T) = rmsnorm(output column (dv) of each (head, token))
// * weight * silu(z), z = proj[z_row + head*dv + row, token]. One warp per
// (head, token); output has leading dimension dv per head (dv, heads, T).
extern "C" __global__ void gd_rms_gate(
    float* gated, const float* output, const float* weight, const float* proj, int ld_proj,
    int z_row, int dv, int heads, float eps) {
    int lane = threadIdx.x;
    int column = blockIdx.x;
    int head = column % heads;
    int token = column / heads;
    const float* o = output + (long long)dv * column;
    float total = 0.0f;
    for (int r = lane; r < dv; r += 32) total += o[r] * o[r];
    float scale = 1.0f / sqrtf(gd_warp_sum(total) / (float)dv + eps);
    for (int r = lane; r < dv; r += 32) {
        float z = proj[z_row + head * dv + r + (long long)ld_proj * token];
        gated[(long long)dv * column + r] = o[r] * scale * weight[r] * gd_silu(z);
    }
}
