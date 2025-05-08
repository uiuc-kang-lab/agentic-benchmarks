#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Store frequently accessed constant 1.0f in constant memory
__constant__ float c_one = 1.0f;

// Kernel using grid-stride loop and constant memory for read-only data
__global__ void swish_const_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = index; i < n; i += stride) {
        float val = x[i];
        // Use constant memory c_one for frequently accessed constant value
        float sig = c_one / (c_one + expf(-val));
        y[i] = val * sig;
    }
}

// Forward function for swish activation using constant memory
torch::Tensor swish_const_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    swish_const_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_const_forward, "Swish activation forward pass using constant memory (CUDA)");
}
