#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK (BLOCK_SIZE/WARP_SIZE)

__global__ void softsign_kernel_shared_opt(const float* __restrict__ x, 
                                         float* __restrict__ out, 
                                         const int num_elements) {
    __shared__ float shared_data[BLOCK_SIZE];
    
    const unsigned int tid = threadIdx.x;
    const unsigned int wid = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;
    const unsigned int gid = blockIdx.x * blockDim.x + tid;
    
    // Cooperative loading into shared memory using warps
    if (gid < num_elements) {
        shared_data[tid] = x[gid];
    }
    __syncthreads();
    
    // Process data in warps
    if (gid < num_elements) {
        // Each warp processes its portion of shared memory
        float val = shared_data[tid];
        
        // Compute softsign - compiler can optimize this better when separated
        float abs_val = fabsf(val);
        float denom = 1.0f + abs_val;
        float result = val / denom;
        
        // Write result directly to global memory
        out[gid] = result;
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
    softsign_kernel_shared_opt<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        num_elements
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-optimized Softsign activation (CUDA)");
}