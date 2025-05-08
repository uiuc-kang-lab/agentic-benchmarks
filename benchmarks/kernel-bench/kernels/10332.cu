#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device function for GELU activation
__device__ __forceinline__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = (x + coeff * x_cubed) * sqrt_2_over_pi;
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void gelu_kernel_tuned(const float* __restrict__ x, float* __restrict__ y, int n) {
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) {
        y[gid] = gelu(x[gid]);
    }
}

// Host function to launch the tuned kernel
torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    auto y = torch::empty_like(x);
    int n = x.numel();

    // Experiment with different block sizes
    const int block_sizes[] = {32, 64, 128, 256, 512};
    int optimal_block_size = 256; // Default to 256
    float min_time = FLT_MAX;

    for (int block_size : block_sizes) {
        int blocks = (n + block_size - 1) / block_size;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        gelu_kernel_tuned<<<blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        if (milliseconds < min_time) {
            min_time = milliseconds;
            optimal_block_size = block_size;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Launch with the optimal block size
    int blocks = (n + optimal_block_size - 1) / optimal_block_size;
    gelu_kernel_tuned<<<blocks, optimal_block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "Tuned GELU forward CUDA implementation with block size optimization");
}