#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device function for GELU activation
__device__ __forceinline__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = (x + coeff * x_cubed) * sqrt_2_over_pi;
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Kernel using vectorized memory accesses to ensure coalescing
__global__ void gelu_kernel_vectorized(const float* __restrict__ x, float* __restrict__ y, int n) {
    // Process the bulk of the data using float4 (vectorized) loads and stores
    int vec_size = n / 4;  // number of float4 elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process vectorized part
    for (int i = idx; i < vec_size; i += stride) {
        // Reinterpret global memory as float4
        float4 in_val = reinterpret_cast<const float4*>(x)[i];
        float4 out_val;
        out_val.x = gelu(in_val.x);
        out_val.y = gelu(in_val.y);
        out_val.z = gelu(in_val.z);
        out_val.w = gelu(in_val.w);
        reinterpret_cast<float4*>(y)[i] = out_val;
    }
    
    // Process any remaining elements
    int remainder = n % 4;
    int tail_start = vec_size * 4;
    for (int j = idx; j < remainder; j += stride) {
        int index = tail_start + j;
        y[index] = gelu(x[index]);
    }
}

// Host function to launch the vectorized kernel
torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto y = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256;
    int num_vec = n / 4;
    int blocks = (num_vec + threads - 1) / threads;
    // Ensure at least one block is launched
    if (blocks == 0) { blocks = 1; }

    gelu_kernel_vectorized<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward CUDA implementation with vectorized memory accesses");
}
