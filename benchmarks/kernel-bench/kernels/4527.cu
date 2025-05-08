#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#define MAX_CHANNELS 2048
#define WARP_SIZE 32

// Constant memory for frequently accessed parameters
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
__global__ void fused_groupnorm_kernel(
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
    
    // Each block handles one (n,g) pair
    const int n = blockIdx.x / num_groups;
    const int g = blockIdx.x % num_groups;
    
    const int group_size = channels_per_group * spatial;
    const int group_offset = n * C * spatial + g * channels_per_group * spatial;
    
    // Shared memory for partial sums and squares
    extern __shared__ char smem[];
    scalar_t* s_sum = reinterpret_cast<scalar_t*>(smem);
    scalar_t* s_sum_sq = s_sum + blockDim.x;
    
    // First pass: compute mean and variance
    scalar_t thread_sum = 0;
    scalar_t thread_sum_sq = 0;
    
    #pragma unroll 4
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        const scalar_t val = x[group_offset + i];
        thread_sum += val;
        thread_sum_sq += val * val;
    }
    
    s_sum[threadIdx.x] = thread_sum;
    s_sum_sq[threadIdx.x] = thread_sum_sq;
    __syncthreads();
    
    // Reduce within block using warp-level primitives when possible
    for (int stride = blockDim.x/2; stride > WARP_SIZE; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Warp-level reduction
    if (threadIdx.x < WARP_SIZE) {
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            s_sum[threadIdx.x] += __shfl_down_sync(0xffffffff, s_sum[threadIdx.x], offset);
            s_sum_sq[threadIdx.x] += __shfl_down_sync(0xffffffff, s_sum_sq[threadIdx.x], offset);
        }
    }
    
    // Store statistics
    if (threadIdx.x == 0) {
        const int stats_idx = n * num_groups + g;
        const scalar_t group_mean = s_sum[0] / group_size;
        const scalar_t group_var = (s_sum_sq[0] / group_size) - (group_mean * group_mean);
        mean[stats_idx] = group_mean;
        var[stats_idx] = group_var;
    }
    __syncthreads();
    
    // Second pass: normalize and apply affine transformation
    const scalar_t group_mean = mean[n * num_groups + g];
    const scalar_t group_var = var[n * num_groups + g];
    const scalar_t inv_std = rsqrt(group_var + eps);
    
    #pragma unroll 4
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        const int offset = group_offset + i;
        const int c = (i / spatial) + g * channels_per_group;
        const scalar_t val = x[offset];
        y[offset] = (val - group_mean) * inv_std * get_weight<scalar_t>(c) + get_bias<scalar_t>(c);
    }
}