#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device function for GELU activation computation
__device__ float compute_gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = x + coeff * x_cubed;
    inner *= sqrt_2_over_pi;
    float tanh_val = tanhf(inner);
    return 0.5f * x * (1.0f + tanh_val);
}

// Optimized kernel with improved thread and block indexing
__global__ void gelu_kernel_optimized_indexing(const float* __restrict__ x, float* __restrict__ y, int n) {
    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int gridSize = gridDim.x * blockSize;
    int idx = blockIdx.x * blockSize + tid;

    // Process elements using improved indexing
    while (idx < n) {
        y[idx] = compute_gelu(x[idx]);
        idx += gridSize;  // Move to the next element this thread should process
    }
}

// Torch binding to launch the optimized GELU kernel
torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    auto y = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    gelu_kernel_optimized_indexing<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "Optimized GELU forward CUDA implementation with improved indexing");
}