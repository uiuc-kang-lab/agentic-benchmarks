#include <torch/extension.h>

__global__ void swish_kernel(const float* x, float* y, int64_t n) {
    const int64_t stride = blockDim.x * gridDim.x;
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop to handle multiple elements per thread
    while (index < n) {
        const float val = x[index];
        const float sigmoid = 1.0f / (1.0f + expf(-val));
        y[index] = val * sigmoid;
        index += stride;
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    
    // Optimize block and grid dimensions for H100
    const int threads = 256;  // Keep thread count optimal for H100
    const int max_blocks = 256;  // Limit blocks for better occupancy
    const int blocks = min((n + threads - 1) / threads, max_blocks);
    
    swish_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish activation forward pass (CUDA)");
}