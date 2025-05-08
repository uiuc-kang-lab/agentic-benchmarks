#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel using tile-based shared memory to cache input values
// Unroll factor = 4 for improved throughput; each block cooperatively loads a tile into shared memory.

__global__ void gelu_kernel_shared_tile(const float* __restrict__ x, float* __restrict__ y, int n) {
    const int unroll_factor = 4;
    int tileSize = blockDim.x * unroll_factor;
    int base = blockIdx.x * tileSize;
    
    // Declare dynamic shared memory for the tile
    extern __shared__ float tile[];
    
    int tid = threadIdx.x;

    // Load a tile of data from global memory into shared memory cooperatively
    #pragma unroll
    for (int i = 0; i < unroll_factor; i++) {
        int idx = base + tid + i * blockDim.x;
        if (idx < n) {
            tile[tid + i * blockDim.x] = x[idx];
        }
    }
    __syncthreads();

    // Constants for the GELU computation
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    // Compute GELU activation using the shared memory tile and write results back to global memory
    #pragma unroll
    for (int i = 0; i < unroll_factor; i++) {
        int idx = base + tid + i * blockDim.x;
        if (idx < n) {
            float xi = tile[tid + i * blockDim.x];
            float x_cubed = xi * xi * xi;
            float inner = (xi + coeff * x_cubed) * sqrt_2_over_pi;
            float tanh_val = tanhf(inner);
            y[idx] = 0.5f * xi * (1.0f + tanh_val);
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
    const int unroll_factor = 4;
    int tileSize = threads * unroll_factor;
    int blocks = (n + tileSize - 1) / tileSize;

    // Allocate shared memory: one tile per block
    size_t sharedMemSize = tileSize * sizeof(float);
    
    gelu_kernel_shared_tile<<<blocks, threads, sharedMemSize>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward CUDA implementation using shared memory tiles");
}
