#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel to process input in float4 vectorized chunks
__global__ void gelu_kernel_vector(const float4* __restrict__ x, float4* __restrict__ y, int vec_size) {
    // Move constants outside thread scope to reduce register pressure
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < vec_size) {
        float4 v = __ldg(&x[i]);
        float4 out;
        
        // Reuse variables to reduce register pressure
        float x_val, x_squared, inner;
        
        // Process x component
        x_val = v.x;
        x_squared = x_val * x_val;
        inner = (x_val + coeff * x_val * x_squared) * sqrt_2_over_pi;
        out.x = 0.5f * x_val * (1.0f + tanhf(inner));
        
        // Process y component
        x_val = v.y;
        x_squared = x_val * x_val;
        inner = (x_val + coeff * x_val * x_squared) * sqrt_2_over_pi;
        out.y = 0.5f * x_val * (1.0f + tanhf(inner));
        
        // Process z component
        x_val = v.z;
        x_squared = x_val * x_val;
        inner = (x_val + coeff * x_val * x_squared) * sqrt_2_over_pi;
        out.z = 0.5f * x_val * (1.0f + tanhf(inner));
        
        // Process w component
        x_val = v.w;
        x_squared = x_val * x_val;
        inner = (x_val + coeff * x_val * x_squared) * sqrt_2_over_pi;
        out.w = 0.5f * x_val * (1.0f + tanhf(inner));
        
        y[i] = out;
    }
}

// Fallback scalar kernel for remaining elements
__global__ void gelu_kernel_scalar(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;
        float xi = __ldg(&x[i]);
        float x_cubed = xi * xi * xi;
        float inner = (xi + coeff * x_cubed) * sqrt_2_over_pi;
        y[i] = 0.5f * xi * (1.0f + tanhf(inner));
    }
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto y = torch::empty_like(x);
    int n = x.numel();

    // Process most of the tensor with vectorized float4 loads/stores
    int vec_size = n / 4;  // number of float4 vectors
    int remainder = n % 4;

    // Using 64 threads per block for potentially better occupancy
    const int threads = 64;
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
    m.def("forward", &gelu_forward, "GELU vectorized forward CUDA implementation");
}