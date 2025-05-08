#include <torch/extension.h>

__global__ void swish_kernel_vectorized(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    // Process 4 elements per thread using float4
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;
    const int64_t vec_stride = stride * 4;
    const int64_t vec_n = n / 4;

    // Main vectorized loop
    for (int64_t i = tid; i < vec_n; i += stride) {
        float4 in_vec = reinterpret_cast<const float4*>(x)[i];
        float4 out_vec;

        // Process each component
        out_vec.x = in_vec.x / (1.0f + expf(-in_vec.x)) * in_vec.x;
        out_vec.y = in_vec.y / (1.0f + expf(-in_vec.y)) * in_vec.y;
        out_vec.z = in_vec.z / (1.0f + expf(-in_vec.z)) * in_vec.z;
        out_vec.w = in_vec.w / (1.0f + expf(-in_vec.w)) * in_vec.w;

        // Store result
        reinterpret_cast<float4*>(y)[i] = out_vec;
    }

    // Handle remaining elements
    const int64_t rem_start = vec_n * 4;
    for (int64_t i = rem_start + tid; i < n; i += stride) {
        const float val = x[i];
        const float sigmoid = 1.0f / (1.0f + expf(-val));
        y[i] = val * sigmoid;
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    
    // Optimize thread and block count for H100
    const int threads = 256;
    const int blocks = std::min(65535, (int)((n + threads * 4 - 1) / (threads * 4)));
    
    swish_kernel_vectorized<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Vectorized Swish activation forward pass (CUDA)");
}