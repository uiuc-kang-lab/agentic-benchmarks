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
        float xi = ((float*)&x)[i];
        float x_cubed = xi * xi * xi;
        float inner = sqrt_2_over_pi * (xi + coeff * x_cubed);
        float tanh_val = __tanhf(inner);
        ((float*)&result)[i] = 0.5f * xi * (1.0f + tanh_val);
    }
    return result;
}

__global__ void gelu_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec_stride = blockDim.x * gridDim.x;
    const int vec_n = n / 4;

    // Vectorized processing
    for (int i = tid; i < vec_n; i += vec_stride) {
        float4 val = *reinterpret_cast<const float4*>(x + i*4);
        *reinterpret_cast<float4*>(y + i*4) = gelu_vec4(val);
    }

    // Handle remaining elements
    const int remainder = n - vec_n*4;
    const int scalar_base = vec_n*4;
    for (int i = scalar_base + tid; i < n; i += vec_stride) {
        float xi = x[i];
        float x_cubed = xi * xi * xi;
        float inner = 0.7978845608f * (xi + 0.044715f * x_cubed);
        float tanh_val = __tanhf(inner);
        y[i] = 0.5f * xi * (1.0f + tanh_val);
    }
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "Input must be contiguous CUDA tensor");
    
    auto y = torch::empty_like(x);
    int n = x.numel();

    // H100-optimized parameters
    const int threads = 128;  // 4 warps per block
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    int blocks = min((n + 4*threads - 1) / (4*threads), sm_count * 32);

    gelu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "Optimized GELU forward");
}
