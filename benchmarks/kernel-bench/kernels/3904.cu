#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define WARP_SIZE 32
#define BLOCK_SIZE 256

__global__ void softsign_kernel_warp_optimized(const float* __restrict__ x, 
                                               float* __restrict__ out, 
                                               const int num_elements) {
    extern __shared__ float shared_data[];
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory without divergence
    float val = 0.0f;
    if (gid < num_elements) {
        val = x[gid];
        shared_data[tid] = val;
    }
    __syncthreads();

    // Use shared memory and warp-synchronous programming to avoid divergence
    if (gid < num_elements) {
        val = shared_data[tid];
        out[gid] = val / (1.0f + fabsf(val));
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    const int num_elements = x.numel();
    
    // Calculate grid dimensions
    const int threads = BLOCK_SIZE;
    const int blocks = (num_elements + threads - 1) / threads;
    
    // Launch kernel with shared memory
    softsign_kernel_warp_optimized<<<blocks, threads, threads * sizeof(float)>>> (
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        num_elements
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-divergence optimized Softsign activation (CUDA)");
}