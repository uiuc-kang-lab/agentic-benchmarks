#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define BLOCK_SIZE 256
#define SHARED_MEM_PADDING 32

__global__ void leaky_relu_coalesced_kernel(const float* __restrict__ x, 
                                          float* __restrict__ out,
                                          float negative_slope, 
                                          int n) {
    __shared__ float shared_data[BLOCK_SIZE + SHARED_MEM_PADDING];
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * BLOCK_SIZE + tid;
    const int stride = gridDim.x * BLOCK_SIZE;

    for (int idx = gid; idx < n; idx += stride) {
        shared_data[tid] = x[idx];
        __syncthreads();

        float val = shared_data[tid];
        out[idx] = val > 0.0f ? val : val * negative_slope;
        
        if (idx + stride < n) {
            __syncthreads();
        }
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = BLOCK_SIZE;
    const int max_blocks = 128;
    const int blocks = min((n + threads - 1) / threads, max_blocks);

    leaky_relu_coalesced_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, n);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward with coalesced memory access (CUDA)");
}