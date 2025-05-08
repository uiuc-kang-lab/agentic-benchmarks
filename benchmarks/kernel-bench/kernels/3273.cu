#include <torch/extension.h>

__global__ void swish_block_opt_kernel(const float* __restrict__ x, float* __restrict__ y, const int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Use a grid-stride loop for efficient workload distribution
    while (idx < n) {
        float val = x[idx];
        float sigmoid = 1.0f / (1.0f + expf(-val));
        y[idx] = val * sigmoid;
        idx += stride;
    }
}

torch::Tensor swish_block_opt_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    
    // Optimize block and thread configuration for better workload distribution
    const int threads = 512;
    const int blocks = (n + threads - 1) / threads;

    swish_block_opt_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_block_opt_forward, "Optimized Swish activation with improved block distribution (CUDA)");
}
