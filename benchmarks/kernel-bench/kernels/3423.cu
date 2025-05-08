#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ inline float gelu_function(float x) {
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

// Coalesced memory access kernel using float4
__global__ void gelu_kernel_coalesced(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    const size_t n4) {
    
    // Calculate block-strided index for coalesced access
    const size_t block_stride = gridDim.x * blockDim.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process elements with stride to maintain coalescing within warps
    while (idx < n4) {
        // Load 4 consecutive floats using float4
        float4 in_val = __ldg(&input[idx]);
        
        // Process the four elements
        in_val.x = gelu_function(in_val.x);
        in_val.y = gelu_function(in_val.y);
        in_val.z = gelu_function(in_val.z);
        in_val.w = gelu_function(in_val.w);
        
        // Store result with coalesced access
        output[idx] = in_val;
        
        // Move to next block-strided position
        idx += block_stride;
    }
}

// Coalesced processing for remainder elements
__global__ void gelu_kernel_remainder_coalesced(
    const float* __restrict__ input,
    float* __restrict__ output,
    const size_t offset,
    const size_t numel) {
    
    // Calculate block-strided index for coalesced access
    const size_t block_stride = gridDim.x * blockDim.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    
    // Process remaining elements with stride
    while (idx < numel) {
        output[idx] = gelu_function(__ldg(&input[idx]));
        idx += block_stride;
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
    
    // Configure kernel launch parameters for optimal occupancy
    const int threads_per_block = 256;
    const int max_blocks = 256;  // Adjust based on input size
    const int num_blocks = min(max_blocks, (int)((n4 + threads_per_block - 1) / threads_per_block));
    
    // Launch main vectorized kernel with coalesced access
    if (n4 > 0) {
        gelu_kernel_coalesced<<<num_blocks, threads_per_block>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            n4);
    }
    
    // Handle remainder elements with coalesced access
    if (remainder > 0) {
        const int rem_blocks = min(max_blocks, (int)((remainder + threads_per_block - 1) / threads_per_block));
        gelu_kernel_remainder_coalesced<<<rem_blocks, threads_per_block>>>(
            x.data_ptr<float>(),
            output.data_ptr<float>(),
            n4 * vec_size,
            numel);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA) with coalesced memory access");
}