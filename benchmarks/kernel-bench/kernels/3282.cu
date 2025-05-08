#include <torch/extension.h>

__device__ __forceinline__ float compute_swish_warp(float x) {
    // Direct computation without branches
    const float sigmoid = 1.0f / (1.0f + expf(-x));
    return x * sigmoid;
}

__global__ void warp_aligned_swish_kernel(const float* __restrict__ x, 
                                        float* __restrict__ y,
                                        const int64_t n) {
    // Calculate warp-aligned index
    const int warp_size = 32;
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = gridDim.x * blockDim.x;
    
    // Ensure stride is warp-aligned
    const int aligned_stride = (stride + warp_size - 1) & ~(warp_size - 1);
    
    // Process elements with warp-aligned access
    for (int idx = tid; idx < n; idx += aligned_stride) {
        // Compute swish directly without branches
        const float val = x[idx];
        y[idx] = compute_swish_warp(val);
    }
}

torch::Tensor warp_aligned_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    
    // Ensure block size is multiple of warp size
    const int threads = 256; // 8 warps per block
    const int blocks = (n + threads - 1) / threads;
    
    warp_aligned_swish_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &warp_aligned_forward, "Warp-aligned Swish forward pass (CUDA)");
}