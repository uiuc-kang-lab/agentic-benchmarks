#include <torch/extension.h>

// CUDA kernel that evenly partitions the input among threads
__global__ void balanced_swish_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    // Total number of threads in the grid
    int total_threads = gridDim.x * blockDim.x;
    // Global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute the number of elements per thread (rounded up)
    int chunk = (n + total_threads - 1) / total_threads;
    
    // Determine the start and end index for this thread
    int start = tid * chunk;
    int end = start + chunk;
    if (end > n) end = n;
    
    // Process a contiguous block of elements
    for (int i = start; i < end; i++) {
        float val = x[i];
        float sigmoid = 1.0f / (1.0f + expf(-val));
        y[i] = val * sigmoid;
    }
}

// PyTorch binding for the forward pass
torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();

    // Choose grid dimensions for high occupancy on H100
    const int threads = 256;
    const int blocks = 1024;

    balanced_swish_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Balanced Swish activation forward pass (CUDA)");
}
