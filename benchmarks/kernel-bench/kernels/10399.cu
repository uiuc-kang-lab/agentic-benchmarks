#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* x, float* y, const int n) {
    extern __shared__ float shared_x[];
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    // Each thread processes multiple elements with stride equal to total number of threads
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < n; i += stride) {
        // Load data into shared memory
        shared_x[threadIdx.x] = x[i];
        __syncthreads();

        float xi = shared_x[threadIdx.x];
        float inner = sqrt_2_over_pi * fmaf(coeff, xi * xi * xi, xi);
        float tanh_val = tanhf(inner);
        y[i] = 0.5f * xi * (1.0f + tanh_val);

        __syncthreads();
    }
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    auto y = torch::empty_like(x);
    int n = x.numel();
    
    // Use 128 threads per block for better occupancy
    const int threads = 128; // Aligning to warp size for better performance
    // Calculate optimal number of blocks based on SM count
    int max_blocks = 0;
    cudaDeviceGetAttribute(&max_blocks, cudaDevAttrMultiProcessorCount, 0);
    max_blocks *= 32; // Multiply by 32 for H100 to ensure enough blocks per SM
    int blocks = min((n + threads - 1) / threads, max_blocks);
    
    // Launch kernel with shared memory size equal to block size
    gelu_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward CUDA implementation");
}