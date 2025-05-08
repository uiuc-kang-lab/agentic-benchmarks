#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void relu_kernel_strided(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    // Calculate thread index and stride
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Vector types for efficient memory access
    using vec4_t = float4;
    vec4_t* in_vec = (vec4_t*)input;
    vec4_t* out_vec = (vec4_t*)output;
    const int vec_size = 4;
    const int vec_elements = size / vec_size;
    
    // Main vectorized stride loop
    #pragma unroll 4
    for (int idx = tid; idx < vec_elements; idx += stride) {
        vec4_t val = in_vec[idx];
        
        // Vectorized ReLU operation
        val.x = fmaxf(val.x, 0.0f);
        val.y = max(val.y, 0.0f);
        val.z = max(val.z, 0.0f);
        val.w = max(val.w, 0.0f);
        
        out_vec[idx] = val;
    }
    
    // Handle remaining elements with scalar operations
    const int remaining_start = vec_elements * vec_size;
    if (remaining_start < size) {
        #pragma unroll 4
        for (int idx = remaining_start + tid; idx < size; idx += stride) {
            output[idx] = max(input[idx], static_cast<scalar_t>(0));
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    // Optimize thread and block count for H100
    const int threads = 512;  // Maximum threads per block for optimal occupancy
    const int min_blocks_per_sm = 2;
    const int num_sms = 132;  // H100 has 132 SMs
    const int blocks = std::min(
        (int)((input.numel() / 4 + threads - 1) / threads),
        num_sms * min_blocks_per_sm
    );
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "relu_kernel_strided", ([&] {
        relu_kernel_strided<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward (CUDA)");
}