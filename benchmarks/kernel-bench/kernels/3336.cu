#include <torch/extension.h>

__global__ void unrolled_swish_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;
    const int64_t elements_per_thread = 4;
    
    // Process 4 elements per thread
    for (int64_t base = tid * elements_per_thread; base < n; base += stride * elements_per_thread) {
        float vals[4];
        float results[4];
        
        // Load 4 elements
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int64_t idx = base + i;
            vals[i] = (idx < n) ? x[idx] : 0.0f;
        }
        
        // Compute swish for all 4 elements
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float sigmoid = __fdividef(1.0f, 1.0f + expf(-vals[i]));
            results[i] = vals[i] * sigmoid;
        }
        
        // Store results
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int64_t idx = base + i;
            if (idx < n) {
                y[idx] = results[i];
            }
        }
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
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
    m.def("forward", &swish_forward, "Swish activation forward pass with 4x unrolling (CUDA)");
}