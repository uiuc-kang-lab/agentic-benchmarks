#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Tunable Swish Kernel: Processes input in a grid-stride loop using a block size that can be tuned based on hardware
__global__ void tunable_swish_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float val = x[i];
        float sig = 1.0f / (1.0f + expf(-val));
        y[i] = val * sig;
    }
}

// Forward function that accepts a tunable block size parameter; default value is set to 256 but can be changed to 32, 64, 128, or 512
torch::Tensor tunable_swish_forward(torch::Tensor x, int block_size = 256) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    int blocks = (n + block_size - 1) / block_size;
    
    // Launch the kernel with the tuned block size
    tunable_swish_kernel<<<blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}

// Pybind11 module definition; block_size can be adjusted from Python for experimentation
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &tunable_swish_forward, "Tunable Swish activation forward pass (CUDA)",
          py::arg("x"), py::arg("block_size") = 256);
}
