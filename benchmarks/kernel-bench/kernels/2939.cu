#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constants for tanh approximation stored in constant memory
__constant__ float c_tanh_coeff[8] = {
    1.0f, -1.0f/3.0f, 2.0f/15.0f, -17.0f/315.0f,
    62.0f/2835.0f, -1382.0f/155925.0f, 21844.0f/6081075.0f, -929569.0f/638512875.0f
};

__constant__ float c_range_limit = 3.0f;

template <typename scalar_t>
__device__ __forceinline__ float fast_tanh(float x) {
    // Quick return for values outside main computation range
    if (x > c_range_limit) return 1.0f;
    if (x < -c_range_limit) return -1.0f;
    
    float x2 = x * x;
    float sum = x;
    float term = x;
    
    // Use constant memory coefficients for Taylor series
    #pragma unroll
    for(int i = 1; i < 8; i++) {
        term *= x2;
        sum += term * c_tanh_coeff[i];
    }
    
    return sum;
}

template <typename scalar_t>
__device__ __forceinline__ float4 tanh_vec4(float4 val) {
    float4 result;
    result.x = fast_tanh<scalar_t>(val.x);
    result.y = fast_tanh<scalar_t>(val.y);
    result.z = fast_tanh<scalar_t>(val.z);
    result.w = fast_tanh<scalar_t>(val.w);
    return result;
}

template <typename scalar_t>
__global__ void tanh_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec4_size = size / 4;
    
    // Process 4 elements at a time using float4
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);
    
    // Main vectorized processing loop
    for (int i = idx; i < vec4_size; i += stride) {
        float4 in4 = input4[i];
        output4[i] = tanh_vec4<scalar_t>(in4);
    }
    
    // Handle remaining elements
    const int remaining_start = vec4_size * 4;
    for (int i = remaining_start + idx; i < size; i += stride) {
        output[i] = fast_tanh<scalar_t>(input[i]);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() / 4 + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_optimized", ([&] {
        tanh_kernel_optimized<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward with constant memory optimization (CUDA)");
}