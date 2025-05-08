#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device function for GELU activation
__device__ __forceinline__ float gelu(const float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = (x + coeff * x_cubed) * sqrt_2_over_pi;
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Kernel that uses grid-stride loops with vectorized memory accesses
__global__ void gelu_kernel_grid_vectorized(const float* __restrict__ x, float* __restrict__ y, int n) {
    int num_float4 = n / 4;       // Number of complete float4 groups
    int remainder = n % 4;        // Remaining elements
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process main portion with vectorized float4 loads/stores for coalescing
    for (int i = tid; i < num_float4; i += stride) {
        float4 in_val = reinterpret_cast<const float4*>(x)[i];
        float4 out_val;
        out_val.x = gelu(in_val.x);
        out_val.y = gelu(in_val.y);
        out_val.z = gelu(in_val.z);
        out_val.w = gelu(in_val.w);
        reinterpret_cast<float4*>(y)[i] = out_val;
    }

    // Process any remaining elements that don't fit into a float4
    int offset = num_float4 * 4;
    for (int i = tid; i < remainder; i += stride) {
        int idx = offset + i;
        y[idx] = gelu(x[idx]);
    }
}

// Host function to launch the kernel
torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto y = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256;
    // Determine grid size to cover the whole data evenly via grid-stride loop
    int grid = (n + threads - 1) / threads;

    gelu_kernel_grid_vectorized<<<grid, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "Grid-stride vectorized GELU forward CUDA implementation");
}
