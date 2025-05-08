#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel that minimizes warp divergence by having fully active blocks without per-thread conditionals
// and only applying bounds checking in a possibly partial block.
__global__ void softsign_kernel_uniform(const float* __restrict__ x, float* __restrict__ out, int num_elements, int full_blocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // For blocks that are completely within range, no conditional is needed per thread
    if (blockIdx.x < full_blocks) {
        float val = x[idx];
        out[idx] = val / (1.0f + fabsf(val));
    } else {
        // For the last (partial) block, do bounds checking
        if (idx < num_elements) {
            float val = x[idx];
            out[idx] = val / (1.0f + fabsf(val));
        }
    }
}

// Forward function: calculates grid dimensions such that most blocks are fully active
// and only the last block (if any) does a bounds check, reducing warp divergence.

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    int threads = 1024;
    
    // Compute the number of fully active blocks (each with exactly 'threads' elements)
    int full_blocks = num_elements / threads;
    int remainder = num_elements % threads;
    int blocks = full_blocks + (remainder > 0 ? 1 : 0);
    
    softsign_kernel_uniform<<<blocks, threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), num_elements, full_blocks
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign activation with uniform warp (CUDA)");
}
