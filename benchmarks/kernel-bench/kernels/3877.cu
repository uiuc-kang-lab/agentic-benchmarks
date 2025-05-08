#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel using 2D grid and block indexing
__global__ void softsign_kernel_2d(const float* __restrict__ x, float* __restrict__ out, int num_elements) {
    // Flatten 2D block and grid indices into a linear index
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) +
              threadIdx.y * blockDim.x + threadIdx.x;
    int stride = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
    
    for (; idx < num_elements; idx += stride) {
        float val = x[idx];
        out[idx] = val / (1.0f + fabsf(val));
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    
    // Configure a 2D block: 32x8 threads (256 threads per block)
    dim3 blockDim(32, 8);
    
    // Calculate the total number of blocks needed
    int totalBlocks = (num_elements + 256 - 1) / 256;
    
    // Arrange blocks in a 2D grid for efficient scheduling
    int gridX = static_cast<int>(sqrt(static_cast<double>(totalBlocks)));
    if (gridX * gridX < totalBlocks) {
        gridX++;
    }
    int gridY = (totalBlocks + gridX - 1) / gridX;
    dim3 gridDim(gridX, gridY);
    
    softsign_kernel_2d<<<gridDim, blockDim>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), num_elements
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign activation using 2D grid indexing (CUDA)");
}
