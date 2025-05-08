#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel leverages shared memory to load a tile of input data from global memory
// into faster on-chip memory. Each thread in a block loads one element into shared memory,
// then computes the ELU activation from the shared memory copy, and writes the result back to global memory.

__global__ void elu_kernel_shared(const float* __restrict__ x, float* __restrict__ out, float alpha, int n) {
    // Allocate shared memory for the current block
    extern __shared__ float s_data[];
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;

    // Load data from global memory to shared memory if within bounds
    if (global_idx < n) {
        s_data[local_idx] = x[global_idx];
    }

    __syncthreads();

    // Compute ELU activation from shared memory data and store back to global memory
    if (global_idx < n) {
        float val = s_data[local_idx];
        out[global_idx] = (val > 0) ? val : alpha * (expf(val) - 1);
    }
}

torch::Tensor elu_cuda_shared(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    // Allocate shared memory for one block (each thread loads one float)
    size_t sharedMemSize = threads * sizeof(float);

    elu_kernel_shared<<<blocks, threads, sharedMemSize>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_shared, "ELU activation with shared memory optimization (CUDA)");
}
