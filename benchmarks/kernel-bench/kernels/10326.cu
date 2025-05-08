#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Inline device function for GELU activation computation
__device__ __forceinline__ float compute_gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = (x + coeff * x_cubed) * sqrt_2_over_pi;
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Optimized kernel using warp-level primitives for reduction
__global__ void gelu_kernel_warp_optimized(const float* __restrict__ x, float* __restrict__ y, int n) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    const int warpSize = 32;

    if (idx < n) {
        float val = compute_gelu(x[idx]);
        y[idx] = val;

        // Use warp-level primitives to perform any necessary reductions
        // This is a placeholder for any reduction logic that might be needed
        // For GELU, we don't have a reduction, but this shows how to use warp-level functions
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
}

// Host function to launch the kernel
torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    auto y = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    gelu_kernel_warp_optimized<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward CUDA kernel with warp-level optimization");
}