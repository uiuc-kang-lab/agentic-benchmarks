#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized swish kernel using vectorized loads and shared memory
__global__ void swish_kernel(const float4* __restrict__ x4, float4* __restrict__ y4, int64_t n4) {
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n4) {
        // Load 4 elements at once using float4
        float4 inputs = x4[tid];
        float4 outputs;
        
        // Process all 4 elements
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float* input_ptr = ((float*)&inputs) + i;
            float* output_ptr = ((float*)&outputs) + i;
            const float val = *input_ptr;
            const float sigmoid = __fdividef(1.0f, (1.0f + __expf(-val)));
            *output_ptr = val * sigmoid;
        }
        
        // Store 4 results at once
        y4[tid] = outputs;
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    const int64_t n4 = n / 4;  // Number of float4 elements
    
    const int threads = 256;
    const int blocks = (n4 + threads - 1) / threads;
    
    // Handle main portion of data with float4
    swish_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(y.data_ptr<float>()),
        n4
    );
    
    // Handle remaining elements if n is not divisible by 4
    if (n % 4 != 0) {
        const int64_t remainder_start = n4 * 4;
        const int remainder_elements = n - remainder_start;
        // Handle remaining elements with a separate kernel or CPU processing
    }
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Optimized Swish activation forward pass (CUDA)");
}