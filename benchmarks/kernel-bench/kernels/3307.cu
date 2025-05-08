#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel using grid-stride looping with __restrict__ and __ldg for improved memory coalescing
__global__ void swish_coalesced_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; idx < n; idx += stride) {
        // __ldg is used for read-only data caching
        const float val = __ldg(&x[idx]);
        const float sigmoid = __fdividef(1.0f, 1.0f + expf(-val));
        y[idx] = val * sigmoid;
    }
}

// Torch forward function
torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    swish_coalesced_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish activation forward pass with memory coalescing (CUDA)");
}
