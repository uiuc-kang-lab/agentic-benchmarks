#include <torch/extension.h>

// Combined optimized swish kernel using a grid-stride loop and __restrict__ pointers for better memory access.
// Using __expf is a slight modification compared to expf for increased performance with acceptable precision loss in many ML applications.
__global__ void combined_swish_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    // Compute the thread's global index
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Process multiple elements per thread using grid-stride loop
    for (; idx < n; idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        float val = x[idx];
        // Use __expf for faster, though slightly less precise, exponential calculations
        float sigmoid = 1.0f / (1.0f + __expf(-val));
        y[idx] = val * sigmoid;
    }
}

// The forward function that integrates with PyTorch
torch::Tensor combined_swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    combined_swish_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &combined_swish_forward, "Combined Swish activation forward pass (CUDA)");
}
