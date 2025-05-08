#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel using shared memory to reduce global memory latency
__global__ void elu_kernel_shared_optimized(const float* x, float* out, float alpha, int n) {
    extern __shared__ float shared_data[];  // Shared memory allocation
    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    if (globalIdx < n) {
        shared_data[tid] = x[globalIdx];
    }
    __syncthreads();  // Ensure all threads have loaded their data

    // Perform ELU calculation
    if (globalIdx < n) {
        float val = shared_data[tid];
        out[globalIdx] = (val > 0) ? val : alpha * (expf(val) - 1);
    }
}

// Host function to launch the kernel
torch::Tensor elu_cuda_shared_optimized(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    size_t sharedMemSize = threads * sizeof(float);

    elu_kernel_shared_optimized<<<blocks, threads, sharedMemSize>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_shared_optimized, "Optimized ELU activation with shared memory (CUDA)");
}