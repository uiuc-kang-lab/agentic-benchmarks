#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Vectorized tanh computation for float4
__device__ __forceinline__ float4 tanh_vec4(float4 val) {
    float4 result;
    result.x = tanhf(val.x);
    result.y = tanhf(val.y);
    result.z = tanhf(val.z);
    result.w = tanhf(val.w);
    return result;
}

// Combined kernel for both vectorized and scalar tanh computation
template <typename scalar_t>
__global__ void tanh_kernel_combined(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec4_size = size / 4;
    
    // Process 4 elements at a time using float4
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);
    
    for (int i = idx; i < vec4_size; i += stride) {
        float4 in4 = input4[i];
        output4[i] = tanh_vec4(in4);
    }
    
    // Handle remaining elements
    const int remaining_start = vec4_size * 4;
    for (int i = remaining_start + idx; i < size; i += stride) {
        output[i] = tanhf(input[i]);
    }
}

// Forward function to launch the combined kernel
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() / 4 + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_combined", ([&] {
        tanh_kernel_combined<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward combined (CUDA)");
}