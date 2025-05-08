#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* x, float* y, const int n) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    // Each thread processes multiple elements with stride equal to total number of threads
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < n; i += stride) {
        float xi = x[i];
        float inner = sqrt_2_over_pi * fmaf(coeff, xi * xi * xi, xi);
        float tanh_val = tanhf(inner);
        y[i] = 0.5f * xi * (1.0f + tanh_val);
    }
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    auto y = torch::empty_like(x);
    int n = x.numel();
    
    // Experiment with different block sizes to find the optimal configuration
    const int threads = 512; // Using a larger block size for potential performance gain
    // Calculate optimal number of blocks based on SM count
    int max_blocks = 0;
    cudaDeviceGetAttribute(&max_blocks, cudaDevAttrMultiProcessorCount, 0);
    max_blocks *= 32; // Multiply by 32 for H100 to ensure enough blocks per SM
    int blocks = min((n + threads - 1) / threads, max_blocks);
    
    gelu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward CUDA implementation");
}