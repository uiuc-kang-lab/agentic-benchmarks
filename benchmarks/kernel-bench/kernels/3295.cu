#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel leveraging shared memory to cache input data and reuse it for swish computation
__global__ void swish_shared_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    extern __shared__ float s_data[];  // Shared memory allocation per block
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    // Load a tile of input data from global memory into shared memory
    if (index < n) {
        s_data[tid] = x[index];
    }
    __syncthreads();

    // Compute swish activation: swish(x) = x * sigmoid(x)
    // Using shared memory to reduce repeated global memory access
    if (index < n) {
        float val = s_data[tid];
        float sig = 1.0f / (1.0f + expf(-val));
        s_data[tid] = val * sig;
    }
    __syncthreads();

    // Write the computed result from shared memory back to global memory
    if (index < n) {
        y[index] = s_data[tid];
    }
}

// Forward function for the swish activation leveraging shared memory
// It allocates shared memory per block and ensures proper synchronization
torch::Tensor swish_shared_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    size_t shared_mem_size = threads * sizeof(float);

    swish_shared_kernel<<<blocks, threads, shared_mem_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}

// Pybind11 module definition keeping the same module name as in the reference
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_shared_forward, "Swish activation forward pass using shared memory (CUDA)");
}
