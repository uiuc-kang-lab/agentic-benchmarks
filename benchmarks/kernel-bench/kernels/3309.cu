#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Optimized Swish kernel using a grid-stride loop for improved occupancy
__global__ void swish_kernel_grid(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple elements in a grid-stride loop
    for (; idx < n; idx += stride) {
        float val = x[idx];
        float sigmoid = 1.0f / (1.0f + expf(-val));
        y[idx] = val * sigmoid;
    }
}

// Swish activation forward pass
torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();

    // Launch configuration: 256 threads per block
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // Launch kernel on the current CUDA stream
    swish_kernel_grid<<<blocks, threads, 0, cudaStreamDefault>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish activation forward pass with grid-stride loop (CUDA)");
}
