#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Combined CUDA kernel leveraging shared memory and optimized indexing
__global__ void elu_kernel_combined(const float* x, float* out, float alpha, int n) {
    extern __shared__ float tile[]; // dynamically allocated shared memory
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Load input data from global memory to shared memory in a coalesced manner
    for (int i = idx; i < n; i += stride) {
        tile[threadIdx.x] = x[i];
        __syncthreads(); // Ensure all threads have loaded data to shared memory before processing

        // Compute the ELU activation using data from shared memory
        float val = tile[threadIdx.x];
        out[i] = (val > 0.0f) ? val : alpha * (expf(val) - 1.0f);

        __syncthreads(); // Ensure all threads have written data before next iteration
    }
}

// Interface function called from Python
torch::Tensor elu_cuda_combined(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();
    
    // Use block size of 256 threads
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    // Allocate shared memory per block
    size_t sharedMemSize = threads * sizeof(float);
    
    elu_kernel_combined<<<blocks, threads, sharedMemSize>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_combined, "ELU activation with combined shared memory and optimized indexing (CUDA)");
}
