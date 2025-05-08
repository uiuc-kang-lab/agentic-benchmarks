#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float4 gelu_vec4(float4 x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    float4 result;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float* xi = ((float*)&x) + i;
        float* yi = ((float*)&result) + i;
        float x_cubed = *xi * *xi * *xi;
        float inner = sqrt_2_over_pi * (*xi + coeff * x_cubed);
        float tanh_val = tanhf(inner);
        *yi = 0.5f * *xi * (1.0f + tanh_val);
    }
    return result;
}

__global__ void gelu_kernel(const float* x, float* y, const int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec_stride = stride * 4;
    const int vec_n = n / 4;
    
    // Process 4 elements at a time using float4
    for (int i = tid; i < vec_n; i += stride) {
        float4* x_vec = (float4*)(x + i * 4);
        float4* y_vec = (float4*)(y + i * 4);
        *y_vec = gelu_vec4(*x_vec);
    }
    
    // Handle remaining elements
    const int remain_start = vec_n * 4;
    for (int i = remain_start + tid; i < n; i += stride) {
        float xi = x[i];
        float x_cubed = xi * xi * xi;
        float inner = 0.7978845608f * (xi + 0.044715f * x_cubed);
        float tanh_val = tanhf(inner);
        y[i] = 0.5f * xi * (1.0f + tanh_val);
    }
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    auto y = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = 256;
    const int max_blocks = 1024;
    const int blocks = min((n + threads - 1) / threads, max_blocks);
    
    gelu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward CUDA implementation");
}