#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define BLOCK_SIZE 256
#define SHARED_MEM_SIZE BLOCK_SIZE

__global__ void elu_kernel_shared(const float* __restrict__ x, float* __restrict__ out, float alpha, int n) {
    __shared__ float shared_data[SHARED_MEM_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Load data into shared memory
    if (idx < n) {
        shared_data[tid] = x[idx];
    }
    
    __syncthreads();
    
    // Process data from shared memory
    if (idx < n) {
        float val = shared_data[tid];
        out[idx] = (val > 0) ? val : alpha * (expf(val) - 1);
    }
}

torch::Tensor elu_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = BLOCK_SIZE;
    const int blocks = (n + threads - 1) / threads;
    
    elu_kernel_shared<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda, "ELU activation with shared memory (CUDA)");
}