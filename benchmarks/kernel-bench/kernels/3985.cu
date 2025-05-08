#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void softsign_optimized_kernel(const float* __restrict__ x,
                                        float* __restrict__ out,
                                        const int num_elements) {
    // Grid-stride loop with 128 threads per block
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_threads = blockDim.x;
    const int grid_size = gridDim.x;
    
    // Calculate starting index and stride
    int idx = bid * num_threads + tid;
    const int stride = grid_size * num_threads;
    
    // Process elements with grid stride
    for(; idx < num_elements; idx += stride) {
        float val = __ldg(&x[idx]); // Use __ldg for read-only cache
        out[idx] = val / (1.0f + fabsf(val));
        idx += stride;
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    const int num_elements = x.numel();
    
    // Optimized launch configuration for H100
    constexpr int threads_per_block = 128;  // 4 warps per block
    constexpr int num_blocks = 8192;        // Large grid for better SM utilization
    
    softsign_optimized_kernel<<<num_blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        num_elements
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign with optimized block size (CUDA)");
}