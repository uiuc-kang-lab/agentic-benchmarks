#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void warp_aligned_swish_kernel(const float4* __restrict__ x4, 
                                        float4* __restrict__ y4, 
                                        const int64_t n4) {
    // Calculate warp-aligned index
    const int64_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int64_t num_warps = (gridDim.x * blockDim.x) / 32;
    
    // Process elements in warp-sized chunks
    for (int64_t i = warp_id; i < n4; i += num_warps) {
        // Load 4 elements
        const float4 inputs = x4[i];
        float4 outputs;
        
        // Process elements - no branches within computation
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float val = ((const float*)&inputs)[j];
            float sigmoid = __fdividef(1.0f, (1.0f + __expf(-val)));
            ((float*)&outputs)[j] = val * sigmoid;
        }
        
        // Store results
        y4[i] = outputs;
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    const int64_t n4 = n / 4;
    
    // Use multiple of 32 (warp size) for thread count
    const int threads = 256; // 8 warps per block
    const int blocks = std::min(65535, (int)((n4 + threads - 1) / threads));
    
    warp_aligned_swish_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(y.data_ptr<float>()),
        n4
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Warp-aligned Swish forward pass (CUDA)");
}