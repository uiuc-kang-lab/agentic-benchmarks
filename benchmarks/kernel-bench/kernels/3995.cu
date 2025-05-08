#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA kernel leveraging shared memory to preload a tile of input data
__global__ void elu_kernel_shared(const float* x, float* out, float alpha, int n) {
    extern __shared__ float tile[]; // dynamically allocated shared memory
    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + tid;

    // Load input data from global memory to shared memory if within bounds
    if (globalIdx < n) {
        tile[tid] = x[globalIdx];
    }
    __syncthreads(); // Ensure all threads have loaded data to shared memory before processing

    // Compute the ELU activation using data from shared memory
    if (globalIdx < n) {
        float val = tile[tid];
        out[globalIdx] = (val > 0.0f) ? val : alpha * (expf(val) - 1.0f);
    }
}

// Interface function called from Python
torch::Tensor elu_cuda_shared(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();
    
    // Use block size of 256 threads
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    // Allocate shared memory per block
    size_t sharedMemSize = threads * sizeof(float);
    
    elu_kernel_shared<<<blocks, threads, sharedMemSize>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_shared, "ELU activation with shared memory (CUDA)");
}
