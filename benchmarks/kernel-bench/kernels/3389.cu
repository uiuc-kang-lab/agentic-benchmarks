#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Explicit specializations of gelu_function for float
template <typename scalar_t>
__device__ inline scalar_t gelu_function(scalar_t x) {
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

// Vectorized CUDA kernel that applies the GELU activation element-wise
__global__ void gelu_kernel_vectorized(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    size_t n4) {
    
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n4) {
        float4 in4 = input[idx];
        
        // Process all four elements
        in4.x = gelu_function(in4.x);
        in4.y = gelu_function(in4.y);
        in4.z = gelu_function(in4.z);
        in4.w = gelu_function(in4.w);
        
        output[idx] = in4;
    }
}

// Handle remaining elements
__global__ void gelu_kernel_remainder(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t offset,
    size_t numel) {
    
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx + offset < numel) {
        output[idx] = gelu_function(input[idx]);
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::ScalarType::Float,
               "Only float32 is supported for vectorized version");
    
    auto output = torch::empty_like(x);
    const size_t numel = x.numel();
    const size_t vec_size = 4;
    const size_t n4 = numel / vec_size;
    const size_t remainder = numel % vec_size;
    
    const int threads = 256;
    const int blocks = (n4 + threads - 1) / threads;
    
    // Main vectorized kernel
    gelu_kernel_vectorized<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(output.data_ptr<float>()),
        n4);
    
    // Handle remaining elements if any
    if (remainder > 0) {
        const int rem_blocks = (remainder + threads - 1) / threads;
        gelu_kernel_remainder<<<rem_blocks, threads>>>(
            x.data_ptr<float>() + n4 * vec_size,
            output.data_ptr<float>() + n4 * vec_size,
            n4 * vec_size,
            numel);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA)");
}