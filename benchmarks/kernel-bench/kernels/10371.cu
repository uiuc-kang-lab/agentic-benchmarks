#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device inline function to compute GELU activation.
__device__ inline float gelu_activation(float x) {
    const float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/pi)
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = (x + coeff * x_cubed) * sqrt_2_over_pi;
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Kernel to process input in float4 vectorized chunks
__global__ void gelu_kernel_vector(const float4* __restrict__ x, float4* __restrict__ y, int vec_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < vec_size) {
        float4 v = __ldg(&x[i]);
        v.x = gelu_activation(v.x);
        v.y = gelu_activation(v.y);
        v.z = gelu_activation(v.z);
        v.w = gelu_activation(v.w);
        y[i] = v;
    }
}

// Fallback scalar kernel for remaining elements
__global__ void gelu_kernel_scalar(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = gelu_activation(x[i]);
    }
}

// Forward function accessible from Python
torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto y = torch::empty_like(x);
    int n = x.numel();

    // Process most of the tensor with vectorized float4 loads/stores
    int vec_size = n / 4;  // number of float4 vectors
    int remainder = n % 4;

    const int threads = 256;
    if(vec_size > 0) {
        int blocks = (vec_size + threads - 1) / threads;
        const float4* x_vec = reinterpret_cast<const float4*>(x.data_ptr<float>());
        float4* y_vec = reinterpret_cast<float4*>(y.data_ptr<float>());
        gelu_kernel_vector<<<blocks, threads>>>(x_vec, y_vec, vec_size);
    }

    // Process any remaining elements with the scalar kernel
    if(remainder > 0) {
        int offset = vec_size * 4;
        int blocks_rem = (remainder + threads - 1) / threads;
        gelu_kernel_scalar<<<blocks_rem, threads>>>(x.data_ptr<float>() + offset, y.data_ptr<float>() + offset, remainder);
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "Optimized GELU combined CUDA implementation");
}
