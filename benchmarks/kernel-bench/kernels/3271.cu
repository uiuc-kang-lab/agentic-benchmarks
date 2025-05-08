#include <torch/extension.h>

// Modular device function for computing sigmoid
__device__ __forceinline__ float compute_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Modular device function for computing single element swish
__device__ __forceinline__ float compute_swish(float x) {
    return x * compute_sigmoid(x);
}

// Modular device function for computing vectorized swish
__device__ __forceinline__ float4 compute_swish_vec4(const float4 vec) {
    float4 result;
    result.x = compute_swish(vec.x);
    result.y = compute_swish(vec.y);
    result.z = compute_swish(vec.z);
    result.w = compute_swish(vec.w);
    return result;
}

__global__ void swish_kernel(const float* __restrict__ x,
                            float* __restrict__ y,
                            const int64_t n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Vector processing
    const int vec_elements = n / 4;
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    float4* y_vec = reinterpret_cast<float4*>(y);
    
    for (int i = tid; i < vec_elements; i += stride) {
        y_vec[i] = compute_swish_vec4(x_vec[i]);
    }
    
    // Handle remaining elements
    const int remainder = vec_elements * 4;
    for (int i = remainder + tid; i < n; i += stride) {
        y[i] = compute_swish(x[i]);
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    swish_kernel<<<blocks, threads>>>(x.data_ptr<float>(),
                                     y.data_ptr<float>(),
                                     n);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Modular vectorized swish forward pass (CUDA)");
}