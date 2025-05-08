#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template<int BLOCK_SIZE>
__global__ void leaky_relu_kernel_optimized(const float* __restrict__ x, 
                                          float* __restrict__ out, 
                                          const float negative_slope, 
                                          const int n) {
    const int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE + tid;
    
    // Use vectorized loads where possible
    if (idx + BLOCK_SIZE <= n && tid % 4 == 0) {
        float4* x4 = (float4*)&x[idx];
        float4* out4 = (float4*)&out[idx];
        float4 val = *x4;
        
        val.x = val.x > 0 ? val.x : val.x * negative_slope;
        val.y = val.y > 0 ? val.y : val.y * negative_slope;
        val.z = val.z > 0 ? val.z : val.z * negative_slope;
        val.w = val.w > 0 ? val.w : val.w * negative_slope;
        
        *out4 = val;
    } else {
        // Handle remaining elements
        if (idx < n) {
            out[idx] = x[idx] > 0 ? x[idx] : x[idx] * negative_slope;
        }
    }
}

torch::Tensor leaky_relu_forward_optimized(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    const int n = x.numel();
    
    constexpr int BLOCK_SIZE = 512;  // Increased block size for better occupancy
    const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    leaky_relu_kernel_optimized<BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        negative_slope, 
        n
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward_optimized, "LeakyReLU forward optimized (CUDA)");
}