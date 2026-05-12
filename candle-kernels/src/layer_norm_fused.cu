#define _USE_MATH_DEFINES
#include "cuda_utils.cuh"
#include <stdint.h>

// Fused single-pass LayerNorm using Welford's online algorithm.
//
// Replaces candle's three-kernel approach (mean → variance → normalize+scale+shift)
// with a single block-per-row kernel. For GNN hidden dims (D=128) this gives
// 3× fewer kernel launches and better L2 locality.
//
// Grid:  N blocks (one per row)
// Block: min(D, 256) threads
// Smem:  2 * blockDim.x floats  (mean + M2 accumulators)
//
// out[row, d] = (x[row,d] - mean) / sqrt(var + eps) * weight[d] + bias[d]

template <typename T>
__device__ __forceinline__ void layer_norm_fused_kernel(
    const size_t N,
    const size_t D,
    const float eps,
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ out
) {
    extern __shared__ float smem[];  // 2 * blockDim.x floats
    float* smem_mean = smem;
    float* smem_m2   = smem + blockDim.x;

    const int row = blockIdx.x;
    if (row >= (int)N) return;

    const T* x_row   = x   + (size_t)row * D;
    T*       out_row = out + (size_t)row * D;

    // --- Thread-local Welford accumulation over assigned elements ---
    float t_mean = 0.f, t_m2 = 0.f;
    int   t_n    = 0;
    for (int d = (int)threadIdx.x; d < (int)D; d += (int)blockDim.x) {
        float val = (float)x_row[d];
        t_n++;
        float delta = val - t_mean;
        t_mean += delta / (float)t_n;
        t_m2   += delta * (val - t_mean);
    }

    smem_mean[threadIdx.x] = t_mean;
    smem_m2  [threadIdx.x] = t_m2;
    __syncthreads();

    // --- Parallel Welford merge (tree reduction) ---
    // We track per-thread element counts to handle D not a multiple of blockDim.x.
    // Since each thread processed ceiling(D/blockDim.x) elements we store counts
    // in a separate pass.  For simplicity we assume D >= blockDim.x (typical:
    // D=128, blockDim=128); if D < blockDim.x some threads contribute 0 elements.
    // The merge below handles that correctly because count_b==0 gives no change.

    // We need counts for the merge; re-derive them from t_n stored locally.
    // Store t_n in smem_m2 temporarily after the initial store — but we already
    // stored m2 there.  Use a simpler merge that is correct when counts differ:
    //   combined_mean = (a_mean * n_a + b_mean * n_b) / (n_a + n_b)
    //   combined_M2   = a_M2 + b_M2 + delta^2 * n_a * n_b / (n_a + n_b)

    // Re-use a third shared buffer (allocated via dynamic smem sized to 3*bdim).
    float* smem_cnt = smem + 2 * blockDim.x;
    smem_cnt[threadIdx.x] = (float)t_n;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float a_mean = smem_mean[threadIdx.x];
            float b_mean = smem_mean[threadIdx.x + stride];
            float a_m2   = smem_m2  [threadIdx.x];
            float b_m2   = smem_m2  [threadIdx.x + stride];
            float n_a    = smem_cnt [threadIdx.x];
            float n_b    = smem_cnt [threadIdx.x + stride];
            float n_ab   = n_a + n_b;
            if (n_ab > 0.f) {
                float delta = b_mean - a_mean;
                smem_mean[threadIdx.x] = (a_mean * n_a + b_mean * n_b) / n_ab;
                smem_m2  [threadIdx.x] = a_m2 + b_m2 + delta * delta * (n_a * n_b / n_ab);
                smem_cnt [threadIdx.x] = n_ab;
            }
        }
        __syncthreads();
    }

    float row_mean = smem_mean[0];
    float row_var  = smem_m2[0] / (float)D;
    float inv_std  = rsqrtf(row_var + eps);

    for (int d = (int)threadIdx.x; d < (int)D; d += (int)blockDim.x) {
        float norm = ((float)x_row[d] - row_mean) * inv_std;
        out_row[d] = (T)(norm * (float)weight[d] + (float)bias[d]);
    }
}

// Explicit instantiations with extern "C" so the PTX symbols are stable.

extern "C" __global__ void layer_norm_fused_f32(
    const size_t N, const size_t D, const float eps,
    const float* x, const float* weight, const float* bias, float* out
) {
    layer_norm_fused_kernel<float>(N, D, eps, x, weight, bias, out);
}

#if __CUDA_ARCH__ >= 530
extern "C" __global__ void layer_norm_fused_f16(
    const size_t N, const size_t D, const float eps,
    const __half* x, const __half* weight, const __half* bias, __half* out
) {
    layer_norm_fused_kernel<__half>(N, D, eps, x, weight, bias, out);
}
#endif

#if __CUDA_ARCH__ >= 800
extern "C" __global__ void layer_norm_fused_bf16(
    const size_t N, const size_t D, const float eps,
    const __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* bias, __nv_bfloat16* out
) {
    layer_norm_fused_kernel<__nv_bfloat16>(N, D, eps, x, weight, bias, out);
}
#endif
