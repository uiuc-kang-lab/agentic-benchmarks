#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ __forceinline__ float4 softsign_vec4(float4 input) {
    float4 result;
    result.x = input.x / (1.0f + fabsf(input.x));
    result.y = input.y / (1.0f + fabsf(input.y));
    result.z = input.z / (1.0f + fabsf(input.z));
    result.w = input.w / (1.0f + fabsf(input.w));
    return result;
}

__global__ void softsign_kernel_vec4(const float* __restrict__ x, float* __restrict__ out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements at a time
    int vec4_elements = num_elements / 4;
    for (int i = idx; i < vec4_elements; i += stride) {
        // Load 4 elements at once
        float4 input = reinterpret_cast<const float4*>(x)[i];
        
        // Process all 4 elements
        float4 result = softsign_vec4(input);
        
        // Store 4 elements at once
        reinterpret_cast<float4*>(out)[i] = result;
    }
    
    // Handle remaining elements
    int remaining_start = vec4_elements * 4;
    for (int i = remaining_start + idx; i < num_elements; i += stride) {
        float val = x[i];
        out[i] = val / (1.0f + fabsf(val));
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    
    // Optimize block size for H100
    int threads = 256;
    int blocks = std::min(65535, (num_elements + threads - 1) / threads);
    
    softsign_kernel_vec4<<<blocks, threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), num_elements
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized Softsign activation (CUDA)");
}