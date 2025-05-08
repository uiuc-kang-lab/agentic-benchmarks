#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Device function for computing the sigmoid
__device__ inline float compute_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Device function for computing the swish activation
__device__ inline float compute_swish(float x) {
    return x * compute_sigmoid(x);
}

// Optimized Swish kernel using atomic operations where necessary
__global__ void atomic_swish_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    extern __shared__ float shared_data[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    if (idx < n) {
        shared_data[tid] = x[idx];
    }
    __syncthreads();

    for (int i = idx; i < n; i += stride) {
        float val = shared_data[tid];
        float swish_val = compute_swish(val);
        atomicAdd(&y[i], swish_val);
    }
}

// CUDA forward function that validates tensor is on device and launches the kernel
torch::Tensor atomic_swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::zeros_like(x); // Initialize with zeros for atomicAdd
    int64_t n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem_size = threads * sizeof(float);
    
    atomic_swish_kernel<<<blocks, threads, shared_mem_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    
    return y;
}

// Pybind11 binding to expose the CUDA function
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &atomic_swish_forward, "Optimized Swish activation with atomic operations (CUDA)");
}
