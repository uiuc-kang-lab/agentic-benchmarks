#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// This kernel uses a grid-stride loop to ensure that the entire input tensor is processed
// even if we launch fewer blocks than there are elements. This can improve occupancy and
// flexibility in mapping threads for different tensor sizes.
__global__ void gelu_kernel(const float* x, float* y, int n) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    // Use grid-stride loop for efficient mapping of threads to problem domain
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float xi = x[i];
        float x_cubed = xi * xi * xi;
        float inner = xi + coeff * x_cubed;
        inner *= sqrt_2_over_pi;
        float tanh_val = tanhf(inner);
        y[i] = 0.5f * xi * (1.0f + tanh_val);
    }
}

// A forward function that launches the kernel with a fixed number of threads per block.
// We compute the number of blocks based on the tensor size and limit it to a reasonable
// number to let the grid-stride loop cover any remaining elements.

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto y = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256; // Aligning block dimensions to warp boundaries (multiple of 32)
    // Launch a moderate number of blocks; the grid-stride loop ensures full coverage
    // even if n is not evenly divisible. Adjust blocks if needed for your workload.
    int blocks = (n + threads - 1) / threads;
    // Optionally, cap the number of blocks to a maximum for improved occupancy
    int maxBlocks = 1024;
    if (blocks > maxBlocks) {
        blocks = maxBlocks;
    }

    gelu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "Grid-stride GELU forward CUDA implementation");
}
