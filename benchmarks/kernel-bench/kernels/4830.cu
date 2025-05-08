#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32
#define EPSILON 1e-12f

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void l1_norm_forward_kernel(const float* __restrict__ x,
                                     float* __restrict__ out,
                                     const int N,
                                     const int D) {
    extern __shared__ float s_partial_sums[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    
    // Compute partial sums with coalesced memory access
    float thread_sum = 0.0f;
    #pragma unroll 4
    for (int col = tid; col < D; col += blockDim.x) {
        thread_sum += fabsf(x[row * D + col]);
    }
    
    // Warp-level reduction
    float warp_sum = warp_reduce_sum(thread_sum);
    
    // First thread in each warp writes the result
    if (lane == 0) {
        s_partial_sums[wid] = warp_sum;
    }
    __syncthreads();
    
    // Single warp handles final reduction
    float final_sum;
    if (tid < warps_per_block) {
        float warp_val = (tid < warps_per_block) ? s_partial_sums[tid] : 0.0f;
        warp_val = warp_reduce_sum(warp_val);
        
        // Avoid branching for epsilon handling
        final_sum = (tid == 0) ? fmaxf(warp_val, EPSILON) : warp_val;
        s_partial_sums[0] = final_sum;
    }
    __syncthreads();
    
    final_sum = s_partial_sums[0];
    const float inv_sum = 1.0f / final_sum;
    
    // Normalize with coalesced memory access
    #pragma unroll 4
    for (int col = tid; col < D; col += blockDim.x) {
        out[row * D + col] = x[row * D + col] * inv_sum;
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(x.dim() == 2, "Expected 2D tensor.");
    x = x.contiguous();
    
    const int N = x.size(0);
    const int D = x.size(1);
    
    auto out = torch::empty_like(x);
    
    const int thread_count = (D < 512) ? 256 : 512;
    const int warps_per_block = thread_count / WARP_SIZE;
    const int shared_mem_size = warps_per_block * sizeof(float);
    
    l1_norm_forward_kernel<<<N, thread_count, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        D
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L1 Normalization forward pass (CUDA)");
}