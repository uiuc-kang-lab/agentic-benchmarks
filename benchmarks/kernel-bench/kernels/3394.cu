#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Define float8 structure for wider vectorization
struct float8 {
    float4 low;
    float4 high;
};

// Explicit GELU function
__device__ inline float gelu_function(float x) {
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

// Vectorized CUDA kernel that applies the GELU activation element-wise
__global__ void gelu_kernel_vectorized(
    const float8* __restrict__ input,
    float8* __restrict__ output,
    size_t n8) {
    
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n8) {
        // Load 8 elements at once
        float8 in8 = input[idx];
        
        // Process first float4
        in8.low.x = gelu_function(in8.low.x);
        in8.low.y = gelu_function(in8.low.y);
        in8.low.z = gelu_function(in8.low.z);
        in8.low.w = gelu_function(in8.low.w);
        
        // Process second float4
        in8.high.x = gelu_function(in8.high.x);
        in8.high.y = gelu_function(in8.high.y);
        in8.high.z = gelu_function(in8.high.z);
        in8.high.w = gelu_function(in8.high.w);
        
        // Store result
        output[idx] = in8;
    }
}

// Handle remaining elements without synchronization
__global__ void gelu_kernel_remainder(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t offset,
    size_t numel) {
    
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    
    // Process multiple elements per thread without synchronization
    for (size_t i = idx + offset; i < numel; i += stride) {
        output[i - offset] = gelu_function(input[i - offset]);
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::ScalarType::Float,
               "Only float32 is supported for vectorized version");
    
    auto output = torch::empty_like(x);
    const size_t numel = x.numel();
    const size_t vec_size = 8;
    const size_t n8 = numel / vec_size;
    const size_t remainder = numel % vec_size;
    
    const int threads = 256;
    const int blocks = (n8 + threads - 1) / threads;
    
    // Main vectorized kernel
    gelu_kernel_vectorized<<<blocks, threads>>>(
        reinterpret_cast<const float8*>(x.data_ptr<float>()),
        reinterpret_cast<float8*>(output.data_ptr<float>()),
        n8);
    
    // Handle remaining elements if any
    if (remainder > 0) {
        const int rem_blocks = (remainder + threads - 1) / threads;
        gelu_kernel_remainder<<<rem_blocks, threads>>>(
            x.data_ptr<float>() + n8 * vec_size,
            output.data_ptr<float>() + n8 * vec_size,
            n8 * vec_size,
            numel);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA)");
}