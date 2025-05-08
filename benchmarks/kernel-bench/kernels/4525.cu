#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#define MAX_CHANNELS 2048
__constant__ float const_weight_float[MAX_CHANNELS];
__constant__ float const_bias_float[MAX_CHANNELS];
__constant__ double const_weight_double[MAX_CHANNELS];
__constant__ double const_bias_double[MAX_CHANNELS];

template <typename scalar_t>
__device__ __forceinline__ scalar_t get_weight(int idx) {
    return std::is_same<scalar_t, float>::value ? const_weight_float[idx] : const_weight_double[idx];
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t get_bias(int idx) {
    return std::is_same<scalar_t, float>::value ? const_bias_float[idx] : const_bias_double[idx];
}

template <typename scalar_t>
__global__ void fused_group_norm_kernel(
    const scalar_t* __restrict__ x,
    const int N,
    const int C,
    const int spatial,
    const int channels_per_group,
    const int num_groups,
    const scalar_t eps,
    scalar_t* __restrict__ y,
    scalar_t* __restrict__ mean,
    scalar_t* __restrict__ var) {

    const int n = blockIdx.x / num_groups;
    const int g = blockIdx.x % num_groups;
    
    const int group_offset = n * C * spatial + g * channels_per_group * spatial;
    const int group_size = channels_per_group * spatial;

    extern __shared__ char smem[];
    scalar_t* s_sum = reinterpret_cast<scalar_t*>(smem);
    scalar_t* s_sum_sq = s_sum + blockDim.x;

    scalar_t thread_sum = 0;
    scalar_t thread_sum_sq = 0;

    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        const scalar_t val = x[group_offset + i];
        thread_sum += val;
        thread_sum_sq += val * val;
    }

    s_sum[threadIdx.x] = thread_sum;
    s_sum_sq[threadIdx.x] = thread_sum_sq;
    __syncthreads();

    if (threadIdx.x < 32) {
        for (int offset = blockDim.x/2; offset > 32; offset >>= 1) {
            if (threadIdx.x < offset) {
                s_sum[threadIdx.x] += s_sum[threadIdx.x + offset];
                s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + offset];
            }
            __syncthreads();
        }

        scalar_t warp_sum = s_sum[threadIdx.x];
        scalar_t warp_sum_sq = s_sum_sq[threadIdx.x];
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
            warp_sum_sq += __shfl_down_sync(0xffffffff, warp_sum_sq, offset);
        }

        if (threadIdx.x == 0) {
            const int stats_idx = n * num_groups + g;
            const scalar_t group_mean = warp_sum / group_size;
            const scalar_t group_var = warp_sum_sq / group_size - group_mean * group_mean;
            mean[stats_idx] = group_mean;
            var[stats_idx] = group_var;
        }
    }
    __syncthreads();

    const int stats_idx = n * num_groups + g;
    const scalar_t group_mean = mean[stats_idx];
    const scalar_t group_var = var[stats_idx];
    const scalar_t inv_std = rsqrt(group_var + eps);

    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        const int c = (i / spatial) + g * channels_per_group;
        const int global_idx = group_offset + i;
        const scalar_t val = x[global_idx];
        y[global_idx] = (val - group_mean) * inv_std * get_weight<scalar_t>(c) + get_bias<scalar_t>(c);
    }
}