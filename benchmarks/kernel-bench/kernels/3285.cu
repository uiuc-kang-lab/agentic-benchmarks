#include <torch/extension.h>

__device__ __forceinline__ float compute_swish(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void unrolled_swish_kernel(const float* __restrict__ x, 
                                    float* __restrict__ y, 
                                    const int64_t n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements per iteration
    const int64_t unroll_stride = stride * 4;
    int64_t i = tid * 4;
    
    #pragma unroll
    while (i + 3 < n) {
        float x0 = x[i];
        float x1 = x[i + 1];
        float x2 = x[i + 2];
        float x3 = x[i + 3];
        
        y[i] = compute_swish(x0);
        y[i + 1] = compute_swish(x1);
        y[i + 2] = compute_swish(x2);
        y[i + 3] = compute_swish(x3);
        
        i += unroll_stride;
    }
    
    // Handle remaining elements
    while (i < n) {
        y[i] = compute_swish(x[i]);
        i += stride;
    }
}

torch::Tensor unrolled_swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    
    const int threads = 256;
    const int blocks = (n + threads * 4 - 1) / (threads * 4);
    
    unrolled_swish_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &unrolled_swish_forward, "Unrolled Swish forward pass (CUDA)");
}