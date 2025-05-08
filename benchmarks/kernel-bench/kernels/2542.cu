#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for ReLU activation with coalesced memory access
template <typename scalar_t>
__global__ void relu_kernel_coalesced(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    // Use float4 for vectorized memory access
    using float4_t = float4;
    float4_t* in4 = (float4_t*)input;
    float4_t* out4 = (float4_t*)output;
    
    // Calculate number of float4 elements
    const int64_t idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride4 = gridDim.x * blockDim.x;
    const int64_t size4 = size / 4;
    
    // Process 4 elements at a time using vectorized load/store
    for (int64_t i = idx4; i < size4; i += stride4) {
        float4_t val4 = in4[i];
        
        // Apply ReLU to each component
        val4.x = val4.x > 0 ? val4.x : 0;
        val4.y = val4.y > 0 ? val4.y : 0;
        val4.z = val4.z > 0 ? val4.z : 0;
        val4.w = val4.w > 0 ? val4.w : 0;
        
        out4[i] = val4;
    }
    
    // Handle remaining elements
    const int64_t remaining_start = size4 * 4;
    for (int64_t i = remaining_start + threadIdx.x + blockIdx.x * blockDim.x;
         i < size;
         i += blockDim.x * gridDim.x) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() + threads * 4 - 1) / (threads * 4);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_kernel_coalesced", ([&] {
        relu_kernel_coalesced<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward with coalesced access (CUDA)");
}