#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float gelu_calc(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = x + coeff * x_cubed;
    inner *= sqrt_2_over_pi;
    float tanh_val = tanhf(inner);
    return 0.5f * x * (1.0f + tanh_val);
}

__global__ void gelu_kernel_minimal_sync(const float4* __restrict__ x4, float4* __restrict__ y4, int n4) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n4) {
        // Load 4 elements at once using float4
        float4 xval = x4[tid];
        
        // Process each component
        float4 yval;
        yval.x = gelu_calc(xval.x);
        yval.y = gelu_calc(xval.y);
        yval.z = gelu_calc(xval.z);
        yval.w = gelu_calc(xval.w);
        
        // Store 4 elements at once
        y4[tid] = yval;
    }
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    auto y = torch::empty_like(x);
    int n = x.numel();
    int n4 = n / 4;  // Number of float4 elements
    
    const int threads = 256;
    int blocks = (n4 + threads - 1) / threads;
    
    gelu_kernel_minimal_sync<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(y.data_ptr<float>()),
        n4
    );
    
    // Handle remaining elements if n is not divisible by 4
    if (n % 4 != 0) {
        int remain_start = n4 * 4;
        for (int i = remain_start; i < n; i++) {
            float xi = x.data_ptr<float>()[i];
            y.data_ptr<float>()[i] = gelu_calc(xi);
        }
    }
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward CUDA implementation with minimal synchronization");
}