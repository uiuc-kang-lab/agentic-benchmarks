#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel that minimizes warp divergence by having fully active blocks without per-thread conditionals
// and only applying bounds checking in a possibly partial block.
__global__ void softsign_kernel_uniform(const float4* __restrict__ x4, float4* __restrict__ out4, int num_elements, int full_blocks) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int idx4 = tid;  // Each thread processes 4 elements
    
    // For blocks that are completely within range
    if (blockIdx.x < full_blocks) {
        float4 val4 = x4[idx4];
        
        // Process all 4 elements
        val4.x = val4.x / (1.0f + fabsf(val4.x));
        val4.y = val4.y / (1.0f + fabsf(val4.y));
        val4.z = val4.z / (1.0f + fabsf(val4.z));
        val4.w = val4.w / (1.0f + fabsf(val4.w));
        
        out4[idx4] = val4;
    } else {
        // For the last (partial) block, do bounds checking
        int base_idx = tid * 4;
        if (base_idx < num_elements) {
            float4 val4 = x4[idx4];
            
            // Process elements with bounds checking
            if (base_idx < num_elements)
                val4.x = val4.x / (1.0f + fabsf(val4.x));
            if (base_idx + 1 < num_elements)
                val4.y = val4.y / (1.0f + fabsf(val4.y));
            if (base_idx + 2 < num_elements)
                val4.z = val4.z / (1.0f + fabsf(val4.z));
            if (base_idx + 3 < num_elements)
                val4.w = val4.w / (1.0f + fabsf(val4.w));
            
            out4[idx4] = val4;
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
