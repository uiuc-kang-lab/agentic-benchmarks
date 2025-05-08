#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void swish_kernel(const float4* __restrict__ x4, float4* __restrict__ y4, int64_t n4) {
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = gridDim.x * blockDim.x;
    
    // Process 8 elements (2 float4s) per thread
    for (int64_t i = tid; i < n4/2; i += stride) {
        // Load first float4
        float4 in1 = x4[i*2];
        float4 out1;
        
        // Manually unroll first float4 computation
        float val0 = ((float*)&in1)[0];
        float val1 = ((float*)&in1)[1];
        float val2 = ((float*)&in1)[2];
        float val3 = ((float*)&in1)[3];
        
        ((float*)&out1)[0] = val0 * __fdividef(1.0f, (1.0f + __expf(-val0)));
        ((float*)&out1)[1] = val1 * __fdividef(1.0f, (1.0f + __expf(-val1)));
        ((float*)&out1)[2] = val2 * __fdividef(1.0f, (1.0f + __expf(-val2)));
        ((float*)&out1)[3] = val3 * __fdividef(1.0f, (1.0f + __expf(-val3)));
        
        // Load second float4
        float4 in2 = x4[i*2 + 1];
        float4 out2;
        
        // Manually unroll second float4 computation
        val0 = ((float*)&in2)[0];
        val1 = ((float*)&in2)[1];
        val2 = ((float*)&in2)[2];
        val3 = ((float*)&in2)[3];
        
        ((float*)&out2)[0] = val0 * __fdividef(1.0f, (1.0f + __expf(-val0)));
        ((float*)&out2)[1] = val1 * __fdividef(1.0f, (1.0f + __expf(-val1)));
        ((float*)&out2)[2] = val2 * __fdividef(1.0f, (1.0f + __expf(-val2)));
        ((float*)&out2)[3] = val3 * __fdividef(1.0f, (1.0f + __expf(-val3)));
        
        // Store results
        y4[i*2] = out1;
        y4[i*2 + 1] = out2;
    }
    
    // Handle remaining float4 if n4 is odd
    if (tid == 0 && (n4 & 1)) {
        int64_t last = n4 - 1;
        float4 in = x4[last];
        float4 out;
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float val = ((float*)&in)[j];
            ((float*)&out)[j] = val * __fdividef(1.0f, (1.0f + __expf(-val)));
        }
        
        y4[last] = out;
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    const int64_t n4 = (n + 3) / 4;  // Round up to nearest float4
    
    const int threads = 256;
    const int blocks = std::min(65535, (int)((n4 + threads - 1) / threads));
    
    swish_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(y.data_ptr<float>()),
        n4
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Unrolled Swish activation forward pass (CUDA)");
}