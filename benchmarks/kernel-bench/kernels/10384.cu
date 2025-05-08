#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device function to compute GELU activation for a scalar value
__device__ __forceinline__ float compute_gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/pi)
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = (x + coeff * x_cubed) * sqrt_2_over_pi;
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Device function to compute GELU activation for a float4 vector
__device__ __forceinline__ float4 compute_gelu_vector(const float4 v) {
    float4 out;
    out.x = compute_gelu(v.x);
    out.y = compute_gelu(v.y);
    out.z = compute_gelu(v.z);
    out.w = compute_gelu(v.w);
    return out;
}

// Kernel to process input in vectorized float4 chunks
__global__ void gelu_kernel_vector(const float4* __restrict__ x, float4* __restrict__ y, int vec_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_size) {
        // Load data using read-only cache
        float4 input = __ldg(&x[idx]);
        // Apply the modular GELU operation
        float4 output = compute_gelu_vector(input);
        y[idx] = output;
    }
}

// Fallback scalar kernel for remaining elements
__global__ void gelu_kernel_scalar(const float* __restrict__ x, float* __restrict__ y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = compute_gelu(x[idx]);
    }
}

// Forward function exposed to Python
torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto y = torch::empty_like(x);
    int n = x.numel();

    // Process the bulk of data using vectorized operations
    int vec_size = n / 4;  // number of float4 vectors
    int remainder = n % 4;

    // Use 128 threads per block for better occupancy balance
// This can lead to more concurrent blocks and better latency hiding
const int threads = 128;
    if (vec_size > 0) {
        int blocks = (vec_size + threads - 1) / threads;
        const float4* x_vec = reinterpret_cast<const float4*>(x.data_ptr<float>());
        float4* y_vec = reinterpret_cast<float4*>(y.data_ptr<float>());
        gelu_kernel_vector<<<blocks, threads>>>(x_vec, y_vec, vec_size);
    }

    // Process any remaining elements with the scalar kernel
    if (remainder > 0) {
        int offset = vec_size * 4;
        int blocks = (remainder + threads - 1) / threads;
        gelu_kernel_scalar<<<blocks, threads>>>(x.data_ptr<float>() + offset, y.data_ptr<float>() + offset, remainder);
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "Modular GELU CUDA implementation");
}
