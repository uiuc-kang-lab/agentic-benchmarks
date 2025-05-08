#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Tunable parameters
#define BLOCK_SIZE 256
#define ITEMS_PER_THREAD 4  // Each thread processes 4 float4 elements
#define TILE_SIZE (BLOCK_SIZE * ITEMS_PER_THREAD)

__device__ __forceinline__ float4 compute_elu(float4 val, float alpha) {
    float4 result;
    result.x = (val.x > 0.f) ? val.x : alpha * (expf(val.x) - 1.f);
    result.y = (val.y > 0.f) ? val.y : alpha * (expf(val.y) - 1.f);
    result.z = (val.z > 0.f) ? val.z : alpha * (expf(val.z) - 1.f);
    result.w = (val.w > 0.f) ? val.w : alpha * (expf(val.w) - 1.f);
    return result;
}

__global__ void elu_kernel_optimized(const float4* __restrict__ x, 
                                   float4* __restrict__ out,
                                   float alpha, 
                                   int n4) {
    __shared__ float4 tile[TILE_SIZE];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    // Each thread processes multiple float4 elements
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        const int idx = bid * TILE_SIZE + tid + i * BLOCK_SIZE;
        if (idx < n4) {
            // Coalesced load from global memory
            float4 val = x[idx];
            // Compute ELU directly and store in shared memory
            tile[tid + i * BLOCK_SIZE] = compute_elu(val, alpha);
        }
    }
    __syncthreads();
    
    // Coalesced write back to global memory
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        const int idx = bid * TILE_SIZE + tid + i * BLOCK_SIZE;
        if (idx < n4) {
            out[idx] = tile[tid + i * BLOCK_SIZE];
        }
    }
}

__global__ void elu_kernel_tail(const float* x, 
                               float* out,
                               float alpha, 
                               int offset, 
                               int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < n) {
        const float val = x[idx];
        out[idx] = (val > 0.f) ? val : alpha * (expf(val) - 1.f);
    }
}

torch::Tensor elu_cuda_optimized(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    
    const int n = x.numel();
    const int n4 = n / 4;
    const int remainder = n % 4;
    
    // Calculate grid size based on items per thread and tile size
    const int blocks = (n4 + TILE_SIZE - 1) / TILE_SIZE;
    
    // Launch optimized kernel for vectorized portion
    if (n4 > 0) {
        elu_kernel_optimized<<<blocks, BLOCK_SIZE>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            alpha,
            n4
        );
    }
    
    // Handle remaining elements
    if (remainder > 0) {
        const int tail_offset = n4 * 4;
        const int tail_blocks = (remainder + BLOCK_SIZE - 1) / BLOCK_SIZE;
        elu_kernel_tail<<<tail_blocks, BLOCK_SIZE>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            alpha,
            tail_offset,
            n
        );
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_optimized, "Optimized ELU activation (CUDA)");
}