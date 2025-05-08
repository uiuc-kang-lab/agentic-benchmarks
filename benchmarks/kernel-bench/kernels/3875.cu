#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void softsign_kernel_aligned(const float* __restrict__ x, float* __restrict__ out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process 4 elements per thread when possible to better utilize memory bandwidth
    int stride = blockDim.x * gridDim.x;
    int aligned_idx = idx * 4;
    
    while (aligned_idx + 3 < num_elements) {
        // Load 4 elements at once using __ldg
        float4 in_val;
        float* in_ptr = (float*)&in_val;
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            in_ptr[i] = __ldg(&x[aligned_idx + i]);
        }
        
        // Process all 4 values
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            in_ptr[i] = in_ptr[i] / (1.0f + fabsf(in_ptr[i]));
        }
        
        // Store results
        *((float4*)&out[aligned_idx]) = in_val;
        
        aligned_idx += stride * 4;
    }
    
    // Handle remaining elements
    while (idx < num_elements) {
        float val = __ldg(&x[idx]);
        out[idx] = val / (1.0f + fabsf(val));
        idx += stride;
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    
    // Adjust block size to maximize occupancy while maintaining efficient vectorized access
    int threads = 256;
    int blocks = std::min(65535, (num_elements + threads * 4 - 1) / (threads * 4));
    
    softsign_kernel_aligned<<<blocks, threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), num_elements
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign activation with aligned memory access (CUDA)");
}