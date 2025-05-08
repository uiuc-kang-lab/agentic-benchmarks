#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void relu_kernel_vectorized(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    // Process 4 elements per thread using vector loads/stores
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec_size = 4;
    
    // Vector types for efficient memory access
    using vec_t = float4;
    vec_t* in_vec = (vec_t*)input;
    vec_t* out_vec = (vec_t*)output;
    const int vec_elements = size / vec_size;
    
    // Main vectorized loop
    for (int i = tid; i < vec_elements; i += stride) {
        vec_t val = in_vec[i];
        
        // Apply ReLU to each component
        val.x = val.x > 0 ? val.x : 0;
        val.y = val.y > 0 ? val.y : 0;
        val.z = val.z > 0 ? val.z : 0;
        val.w = val.w > 0 ? val.w : 0;
        
        out_vec[i] = val;
    }
    
    // Handle remaining elements
    const int remaining_start = vec_elements * vec_size;
    for (int i = remaining_start + tid; i < size; i += stride) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = std::min(256, (int)((input.numel() / 4 + threads - 1) / threads));
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "relu_kernel_vectorized", ([&] {
        relu_kernel_vectorized<scalar_t><<<blocks, threads>>>(
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