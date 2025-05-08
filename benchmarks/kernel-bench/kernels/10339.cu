#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device function for GELU activation using full precision
__device__ __forceinline__ float compute_gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = (x + coeff * x_cubed) * sqrt_2_over_pi;
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Kernel that uses __ldg() to perform aligned, vectorized loads from global memory
__global__ void gelu_kernel_ldg(const float* __restrict__ x, float* __restrict__ y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;

    // Process most elements in groups of 4 (128-bit aligned accesses)
    int n_vec = n / 4;             // Number of float4 elements
    int remainder = n % 4;           // Leftover elements

    // Cast pointers to float4 for 128-bit loads and stores
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    float4* y_vec = reinterpret_cast<float4*>(y);

    // Loop over vectorized portion using __ldg for read-only load
    for (int i = idx; i < n_vec; i += total_threads) {
        // Use __ldg to perform a cached load from global memory
        float4 in_val = __ldg(&x_vec[i]);
        float4 out_val;
        out_val.x = compute_gelu(in_val.x);
        out_val.y = compute_gelu(in_val.y);
        out_val.z = compute_gelu(in_val.z);
        out_val.w = compute_gelu(in_val.w);
        y_vec[i] = out_val;
    }

    // Process any remaining elements that do not form a complete float4
    int base = n_vec * 4;
    for (int i = idx; i < remainder; i += total_threads) {
        int index = base + i;
        y[index] = compute_gelu(x[index]);
    }
}

// Host function to launch the kernel
torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    auto y = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = 256;
    int n_vec = n / 4;
    int blocks = (n_vec + threads - 1) / threads;
    if (blocks < 1) blocks = 1; // Ensure at least one block

    gelu_kernel_ldg<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "LDG-based aligned GELU forward CUDA implementation");
}
