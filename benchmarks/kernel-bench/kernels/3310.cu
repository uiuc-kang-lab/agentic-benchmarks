#include <torch/extension.h>

__global__ void swish_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t index = tid * 4;
    
    if (index < n) {
        float4 x4;
        float4 y4;
        
        if (index + 3 < n) {
            // Vector load when we have full float4 alignment
            x4 = __ldg(reinterpret_cast<const float4*>(x + index));
            
            // Process all elements using the same pattern
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float* val = &reinterpret_cast<float*>(&x4)[i];
                reinterpret_cast<float*>(&y4)[i] = *val * (1.0f / (1.0f + expf(-*val)));
            }
            
            // Vector store
            reinterpret_cast<float4*>(y + index)[0] = y4;
        } else {
            // Handle boundary case
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (index + i < n) {
                    float val = __ldg(&x[index + i]);
                    y[index + i] = val * (1.0f / (1.0f + expf(-val)));
                }
            }
        }
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    const int threads = 256;
    const int blocks = (n + 4 * threads - 1) / (4 * threads);
    
    swish_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish activation forward pass (CUDA)");
}