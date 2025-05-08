#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void leaky_relu_kernel(const float* x, float* out, float negative_slope, int n) {
    extern __shared__ float shared_x[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (idx < n) {
        shared_x[tid] = x[idx];
    }
    __syncthreads();
    
    // Compute LeakyReLU from shared memory
    if (idx < n) {
        float val = shared_x[tid];
        out[idx] = val > 0 ? val : val * negative_slope;
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = 1024;
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem_size = threads * sizeof(float);
    
    leaky_relu_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, n
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward (CUDA)");
}