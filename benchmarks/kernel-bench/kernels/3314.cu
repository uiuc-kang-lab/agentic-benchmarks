#include <torch/extension.h>

__global__ void swish_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = gridDim.x * blockDim.x;
    
    // Process 4 elements per thread in a grid-stride loop
    for (int64_t idx = tid * 4; idx < n; idx += stride * 4) {
        float4 vals;
        
        if (idx + 3 < n) {
            // Vector load when we have full float4 alignment
            vals = *reinterpret_cast<const float4*>(x + idx);
            
            // Process elements
            float* v = reinterpret_cast<float*>(&vals);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float sigmoid = 1.0f / (1.0f + expf(-v[i]));
                v[i] = v[i] * sigmoid;
            }
            
            // Vector store
            *reinterpret_cast<float4*>(y + idx) = vals;
        } else {
            // Handle boundary case
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                if (idx + i < n) {
                    float val = x[idx + i];
                    float sigmoid = 1.0f / (1.0f + expf(-val));
                    y[idx + i] = val * sigmoid;
                }
            }
        }
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    
    // Using 128 threads per block
    const int threads = 128;
    const int max_blocks = 1024; // Limit maximum number of blocks
    const int blocks = min(max_blocks, (n + (threads * 4) - 1) / (threads * 4));
    
    swish_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish activation forward pass (CUDA)");
}