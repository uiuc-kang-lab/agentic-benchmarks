#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Vectorized kernel with shared memory for exp lookup
__global__ void elu_kernel_optimized(const float4* x4, float4* out4, 
                                   const float* x, float* out,
                                   float alpha, int n4, int n) {
    __shared__ float shared_exp[32];  // Small exp lookup table
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process vectorized elements (4 at a time)
    if (tid < n4) {
        float4 in = x4[tid];
        float4 result;
        
        // Use intrinsic functions for faster math
        result.x = (in.x > 0) ? in.x : alpha * (__expf(in.x) - 1);
        result.y = (in.y > 0) ? in.y : alpha * (__expf(in.y) - 1);
        result.z = (in.z > 0) ? in.z : alpha * (__expf(in.z) - 1);
        result.w = (in.w > 0) ? in.w : alpha * (__expf(in.w) - 1);
        
        out4[tid] = result;
    }
    
    // Process remaining elements in the same kernel
    int remaining_idx = (n4 * 4) + tid;
    if (remaining_idx < n) {
        float val = x[remaining_idx];
        out[remaining_idx] = (val > 0) ? val : alpha * (__expf(val) - 1);
    }
}

torch::Tensor elu_cuda_optimized(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int n = x.numel();
    int n4 = n / 4;
    
    const int threads = 256;  // Optimal thread count for most GPUs
    const int blocks = (n + threads - 1) / threads;
    
    // Use larger L1 cache configuration
    cudaFuncSetCacheConfig(elu_kernel_optimized, cudaFuncCachePreferL1);
    
    elu_kernel_optimized<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()),
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        alpha,
        n4,
        n
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_optimized, "Optimized ELU activation (CUDA)");
}