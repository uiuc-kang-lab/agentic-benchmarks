#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void softsign_kernel_minimal_sync(const float* __restrict__ x, 
                                           float* __restrict__ out,
                                           const int num_elements) {
    extern __shared__ float shared_mem[];
    const int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    
    // Direct processing of aligned elements (no shared memory needed)
    while (gid + 3 < num_elements) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float val = x[gid + i];
            out[gid + i] = val / (1.0f + fabsf(val));
        }
        gid += stride;
    }
    
    // Process remaining elements using shared memory for better coalescing
    if (gid < num_elements) {
        shared_mem[tid] = x[gid];
        // Only sync if we're actually using shared memory
        __syncthreads();
        
        float val = shared_mem[tid];
        out[gid] = val / (1.0f + fabsf(val));
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    const int num_elements = x.numel();
    
    // Optimize thread and block count for H100
    const int threads = 256;
    const int blocks = std::min(65535, (num_elements + threads - 1) / threads);
    
    // Only allocate shared memory for the remaining elements portion
    const int shared_mem_size = threads * sizeof(float);
    
    softsign_kernel_minimal_sync<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        num_elements
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Minimal sync Softsign activation (CUDA)");
}