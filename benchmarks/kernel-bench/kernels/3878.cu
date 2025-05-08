#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define WARP_SIZE 32

__global__ void softsign_kernel_warp(const float* __restrict__ x, float* __restrict__ out, int num_elements) {
    const unsigned int tid = threadIdx.x;
    const unsigned int wid = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;
    const unsigned int warp_offset = (blockIdx.x * (blockDim.x / WARP_SIZE) + wid) * WARP_SIZE;
    const unsigned int grid_stride = gridDim.x * (blockDim.x / WARP_SIZE) * WARP_SIZE;

    // Process elements in warp-sized chunks
    for (unsigned int idx = warp_offset + lane; idx < num_elements; idx += grid_stride) {
        float val = x[idx];
        
        // Compute softsign
        float result = val / (1.0f + fabsf(val));
        
        // Use warp shuffle to share results within the warp if needed
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float shuffled = __shfl_down_sync(0xffffffff, result, offset);
            // In this case, we don't need to combine results, but the primitive is available
        }
        
        out[idx] = result;
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    
    // Configure grid and block dimensions for warp-based execution
    int threads_per_block = 256; // Multiple of warp size
    int warps_per_block = threads_per_block / WARP_SIZE;
    int blocks = (num_elements + (threads_per_block - 1)) / threads_per_block;
    
    // Limit blocks to maximum grid size
    blocks = std::min(blocks, 65535);
    
    softsign_kernel_warp<<<blocks, threads_per_block>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), num_elements
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign activation with warp optimization (CUDA)");
}