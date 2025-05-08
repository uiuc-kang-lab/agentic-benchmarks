#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void elu_kernel_optimized(const float* __restrict__ x, float* __restrict__ out, float alpha, int n) {
    __shared__ float tile[1024];  // Shared memory tile for input data
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    const int grid_size = block_size * 4;  // Each thread processes 4 elements
    
    // Base index for this thread block
    const int block_start = bid * grid_size;
    
    // Process data in chunks
    #pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
        const int chunk_offset = chunk * block_size;
        const int global_idx = block_start + chunk_offset + tid;
        
        // Load data into shared memory
        if (global_idx < n) {
            tile[tid] = x[global_idx];
        }
        __syncthreads();
        
        // Process data in shared memory
        if (global_idx < n) {
            float val = tile[tid];
            out[global_idx] = (val > 0) ? val : alpha * (expf(val) - 1);
        }
        __syncthreads();  // Ensure shared memory is ready for next iteration
    }
}

torch::Tensor elu_cuda_optimized(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 512;
    const int blocks = (n + threads * 4 - 1) / (threads * 4);

    elu_kernel_optimized<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_optimized, "Optimized ELU without atomics (CUDA)");
}