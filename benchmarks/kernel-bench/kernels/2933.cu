#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Function to apply tanh to a float4 vector
__device__ __forceinline__ float4 tanh_vec4(float4 val) {
    float4 result;
    result.x = tanhf(val.x);
    result.y = tanhf(val.y);
    result.z = tanhf(val.z);
    result.w = tanhf(val.w);
    return result;
}

// Kernel with manual loop unrolling
__global__ void tanh_kernel_unrolled(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec4_size = size / 4;
    
    // Process 4 elements at a time using float4
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);
    
    // Loop unrolling for better performance
    for (int i = idx; i < vec4_size; i += stride * 4) {
        if (i < vec4_size) output4[i] = tanh_vec4(input4[i]);
        if (i + stride < vec4_size) output4[i + stride] = tanh_vec4(input4[i + stride]);
        if (i + 2 * stride < vec4_size) output4[i + 2 * stride] = tanh_vec4(input4[i + 2 * stride]);
        if (i + 3 * stride < vec4_size) output4[i + 3 * stride] = tanh_vec4(input4[i + 3 * stride]);
    }
    
    // Handle remaining elements
    const int remaining_start = vec4_size * 4;
    for (int i = remaining_start + idx; i < size; i += stride) {
        output[i] = tanhf(input[i]);
    }
}

// Forward function wrapping the kernel launch

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() / 4 + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_unrolled", ([&] {
        tanh_kernel_unrolled<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward unrolled (CUDA)");
}