#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ inline float gelu_function(float x) {
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

// 2D block organization for better memory access patterns
__global__ void gelu_kernel_2d(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int rows,
    const int cols) {
    
    // 2D thread indexing
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Calculate global indices
    const int row = by * blockDim.y + ty;
    const int col = bx * blockDim.x + tx;
    const int idx = row * cols + col;
    
    // Bounds checking
    if (row < rows && col < cols) {
        float val = __ldg(&input[idx]);
        output[idx] = gelu_function(val);
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::ScalarType::Float,
               "Only float32 is supported for this optimized version");
    
    auto output = torch::empty_like(x);
    
    // Get tensor dimensions
    const int numel = x.numel();
    const int cols = x.size(-1);
    const int rows = numel / cols;
    
    // Define 2D block configuration
    const dim3 threads(32, 8);  // 32x8 = 256 threads per block
    const dim3 blocks(
        (cols + threads.x - 1) / threads.x,
        (rows + threads.y - 1) / threads.y
    );
    
    gelu_kernel_2d<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA) with 2D block organization");
}