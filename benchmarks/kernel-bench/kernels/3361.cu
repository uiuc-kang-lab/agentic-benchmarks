#include <torch/extension.h>

__global__ void swish_kernel_vec4(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;
    const int64_t vec4_elements = n / 4;
    
    // Process 4 elements at a time using float4
    float4* x4 = (float4*)x;
    float4* y4 = (float4*)y;
    
    for (int64_t i = idx; i < vec4_elements; i += stride) {
        float4 inputs = x4[i];
        
        // Process each component
        float4 outputs;
        outputs.x = inputs.x / (1.0f + expf(-inputs.x)) * inputs.x;
        outputs.y = inputs.y / (1.0f + expf(-inputs.y)) * inputs.y;
        outputs.z = inputs.z / (1.0f + expf(-inputs.z)) * inputs.z;
        outputs.w = inputs.w / (1.0f + expf(-inputs.w)) * inputs.w;
        
        y4[i] = outputs;
    }
    
    // Handle remaining elements
    const int64_t remaining_start = vec4_elements * 4;
    for (int64_t i = remaining_start + idx; i < n; i += stride) {
        const float val = x[i];
        y[i] = val / (1.0f + expf(-val)) * val;
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads * 4 - 1) / (threads * 4);
    
    swish_kernel_vec4<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Vectorized Swish activation forward pass (CUDA)");
}