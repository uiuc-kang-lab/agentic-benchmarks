#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Modular device function for computing the sigmoid
__device__ inline float compute_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Modular device function for computing the swish activation
__device__ inline float compute_swish(float x) {
    return x * compute_sigmoid(x);
}

// Optimized Swish kernel, minimizing unnecessary synchronizations
__global__ void efficient_swish_no_excess_sync_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    // Use shared memory for caching values if block size is appropriate
    extern __shared__ float shared_x[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    while (idx < n) {
        // Load elements into shared memory if necessary
        if (tid < blockDim.x && idx < n) {
            shared_x[tid] = x[idx];
        }
        __syncthreads(); // Synchronize only after loading data into shared memory

        if (tid < blockDim.x && idx < n) {
            float val = shared_x[tid];
            y[idx] = compute_swish(val);
        }
        __syncthreads(); // Synchronize only after processing

        idx += stride;
    }
}

// CUDA forward function that validates tensor is on device and launches the kernel
torch::Tensor efficient_swish_no_excess_sync_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    efficient_swish_no_excess_sync_kernel<<<blocks, threads, threads * sizeof(float)>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}

// Pybind11 binding to expose the CUDA function
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &efficient_swish_no_excess_sync_forward, "Efficient Swish activation with minimized synchronizations (CUDA)");
}
