#include <torch/extension.h>

__global__ void optimized_swish_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    for (int64_t i = blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
    const int64_t idx = i + threadIdx.x;
    // Load a tile of data into shared memory
    extern __shared__ float smem[];
    float val = (idx < n) ? x[idx] : 0.0f;
    smem[threadIdx.x] = val;
    __syncthreads();
    
    // Process the loaded tile: each thread computes on its element
    if (idx < n) {
        float a = smem[threadIdx.x];
        float sigmoid = __frcp_rn(1.0f + expf(-a));
        y[idx] = a * sigmoid;
    }
    __syncthreads();
}
}

torch::Tensor optimized_swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    optimized_swish_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_swish_forward, "Optimized Swish activation forward pass (CUDA)");
}