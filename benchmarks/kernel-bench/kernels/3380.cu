#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Optimized Swish kernel using __ldg() for read-only memory access
__global__ void swish_ldg_optimized(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < n; idx += stride) {
        float v = __ldg(&x[idx]);  // Use __ldg() for read-only access
        float sig = 1.0f / (1.0f + expf(-v));
        y[idx] = v * sig;
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    const int threads = 256;
    // Calculate initial block count
    int blocks = (n + threads - 1) / threads;
    
    // Limit blocks to 2 blocks/SM instead of 4 for better resource utilization
    // This can help balance occupancy and per-thread resources
    blocks = min(blocks, 288);  // 2 blocks/SM * 144 SMs on H100

    swish_ldg_optimized<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish activation forward pass (CUDA) with __ldg optimization");
}