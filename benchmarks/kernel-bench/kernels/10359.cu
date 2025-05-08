#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Vectorized kernel using stride loop for processing float4 chunks
__global__ void gelu_kernel_vector_stride(const float4* __restrict__ x, float4* __restrict__ y, int vec_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    for (int i = idx; i < vec_size; i += stride) {
        // Load a vector of 4 floats using read-only cache
        float4 v = __ldg(&x[i]);
        
        // Process each component of the vector
        float x0 = v.x;
        float x1 = v.y;
        float x2 = v.z;
        float x3 = v.w;

        float x0_cubed = x0 * x0 * x0;
        float inner0 = (x0 + coeff * x0_cubed) * sqrt_2_over_pi;
        float y0 = 0.5f * x0 * (1.0f + tanhf(inner0));

        float x1_cubed = x1 * x1 * x1;
        float inner1 = (x1 + coeff * x1_cubed) * sqrt_2_over_pi;
        float y1 = 0.5f * x1 * (1.0f + tanhf(inner1));

        float x2_cubed = x2 * x2 * x2;
        float inner2 = (x2 + coeff * x2_cubed) * sqrt_2_over_pi;
        float y2 = 0.5f * x2 * (1.0f + tanhf(inner2));

        float x3_cubed = x3 * x3 * x3;
        float inner3 = (x3 + coeff * x3_cubed) * sqrt_2_over_pi;
        float y3 = 0.5f * x3 * (1.0f + tanhf(inner3));
        
        float4 out;
        out.x = y0;
        out.y = y1;
        out.z = y2;
        out.w = y3;

        y[i] = out;
    }
}

// Scalar fallback kernel using stride loop for remaining elements
__global__ void gelu_kernel_scalar_stride(const float* __restrict__ x, float* __restrict__ y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    for (int i = idx; i < n; i += stride) {
        float xi = __ldg(&x[i]);
        float xi_cubed = xi * xi * xi;
        float inner = (xi + coeff * xi_cubed) * sqrt_2_over_pi;
        y[i] = 0.5f * xi * (1.0f + tanhf(inner));
    }
}

// CUDA forward function accessible from Python using pybind11
torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto y = torch::empty_like(x);
    int size = x.numel();
    
    // Process in vectorized mode in chunks of 4 floats
    int vec_size = size / 4;  // Number of float4 elements
    int remainder = size % 4;

    int threads = 256;
    // Launch configuration: use a grid that covers the entire workload
    int blocks = (vec_size > 0) ? (vec_size + (threads / 4) - 1) / (threads / 4) : 1;

    if (vec_size > 0) {
        const float4* x_vec = reinterpret_cast<const float4*>(x.data_ptr<float>());
        float4* y_vec = reinterpret_cast<float4*>(y.data_ptr<float>());
        gelu_kernel_vector_stride<<<blocks, threads>>>(x_vec, y_vec, vec_size);
    }

    if (remainder > 0) {
        int offset = vec_size * 4;
        gelu_kernel_scalar_stride<<<blocks, threads>>>(x.data_ptr<float>() + offset, y.data_ptr<float>() + offset, remainder);
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU stride loop vectorized forward CUDA implementation");
}
