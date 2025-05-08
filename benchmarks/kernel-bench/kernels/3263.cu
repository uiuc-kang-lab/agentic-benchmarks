#include <torch/extension.h>

__constant__ float constant_x[1024];  // Assuming x can fit into constant memory

__global__ void swish_kernel_constant(const float* x, float* y, int64_t n) {
    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        const float val = constant_x[index];
        const float sigmoid = 1.0f / (1.0f + expf(-val));
        y[index] = val * sigmoid;
    }
}

torch::Tensor swish_forward_constant(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    cudaMemcpyToSymbol(constant_x, x.data_ptr<float>(), n * sizeof(float));
    
    swish_kernel_constant<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_constant", &swish_forward_constant, "Swish activation forward pass with constant memory (CUDA)");
}