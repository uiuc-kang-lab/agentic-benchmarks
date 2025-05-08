#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized device function for processing a single element
template <typename scalar_t>
__device__ __forceinline__ scalar_t process_single(scalar_t val) {
    return tanhf(val);
}

// Optimized device function for processing vector of 4 elements
template <typename scalar_t>
__device__ __forceinline__ float4 process_vector4(float4 val) {
    float4 result;
    result.x = process_single(val.x);
    result.y = process_single(val.y);
    result.z = process_single(val.z);
    result.w = process_single(val.w);
    return result;
}

// Device function to handle vector processing
template <typename scalar_t>
__device__ __forceinline__ void process_vector_elements(
    const float4* __restrict__ input4,
    float4* __restrict__ output4,
    const int idx,
    const int stride,
    const int vec4_size) {
    
    for (int i = idx; i < vec4_size; i += stride) {
        float4 in4 = input4[i];
        output4[i] = process_vector4<scalar_t>(in4);
    }
}

// Device function to handle remaining scalar elements
template <typename scalar_t>
__device__ __forceinline__ void process_remaining_elements(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int idx,
    const int stride,
    const int remaining_start,
    const int size) {
    
    for (int i = remaining_start + idx; i < size; i += stride) {
        output[i] = process_single(input[i]);
    }
}

template <typename scalar_t>
__global__ void tanh_kernel_modular(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec4_size = size / 4;
    
    // Process vector elements
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);
    process_vector_elements<scalar_t>(input4, output4, idx, stride, vec4_size);
    
    // Process remaining elements
    const int remaining_start = vec4_size * 4;
    process_remaining_elements<scalar_t>(input, output, idx, stride, remaining_start, size);
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() / 4 + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_modular", ([&] {
        tanh_kernel_modular<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward modular vectorized (CUDA)");
}