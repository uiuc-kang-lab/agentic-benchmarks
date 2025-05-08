#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

template <typename scalar_t>
__device__ inline scalar_t gelu_function(scalar_t x) {
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

// Vectorized kernel for aligned data processing using float4
__global__ void gelu_kernel_vectorized(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    size_t n4) {
    
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n4) {
        // Use __ldg for better cache behavior
        float4 in4 = __ldg(&input[idx]);
        
        // Process all four elements
        in4.x = gelu_function(in4.x);
        in4.y = gelu_function(in4.y);
        in4.z = gelu_function(in4.z);
        in4.w = gelu_function(in4.w);
        
        output[idx] = in4;
    }
}

// Generic kernel for non-float4 types and remainder elements
template <typename scalar_t>
__global__ void gelu_kernel_generic(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    size_t offset,
    size_t numel) {
    
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    
    // Process multiple elements per thread for better occupancy
    for (size_t i = idx + offset; i < numel; i += stride) {
        scalar_t val = __ldg(&input[i]);
        output[i] = gelu_function(val);
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    
    auto output = torch::empty_like(x);
    const size_t numel = x.numel();
    
    if (x.scalar_type() == torch::ScalarType::Float) {
        // Use vectorized version for float
        const size_t vec_size = 4;
        const size_t n4 = numel / vec_size;
        const size_t remainder = numel % vec_size;
        
        const int threads = 256;
        const int blocks = (n4 + threads - 1) / threads;
        
        // Main vectorized kernel
        if (n4 > 0) {
            gelu_kernel_vectorized<<<blocks, threads>>>(
                reinterpret_cast<const float4*>(x.data_ptr<float>()),
                reinterpret_cast<float4*>(output.data_ptr<float>()),
                n4);
        }
        
        // Handle remaining elements
        if (remainder > 0) {
            const int rem_blocks = (remainder + threads - 1) / threads;
            gelu_kernel_generic<<<rem_blocks, threads>>>(
                x.data_ptr<float>(),
                output.data_ptr<float>(),
                n4 * vec_size,
                numel);
        }
    } else {
        // Generic version for other types
        const int threads = 256;
        const int blocks = (numel + threads - 1) / threads;
        
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_cuda", ([&] {
            gelu_kernel_generic<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                0,
                numel);
        }));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA)");
}