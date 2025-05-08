#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define TILE_SIZE as a compile-time constant to experiment with block sizes.
// Typical values: 16, 32. For H100, TILE_SIZE=32 (yielding 1024 threads/block) is a good candidate.
#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

// CUDA kernel for computing C = A * B where A and B are lower triangular matrices. 
// The computation is done only for the lower triangular part (i.e., if row >= col).
__global__ void triangular_mm_kernel_optimized(const float* __restrict__ A,
                                                const float* __restrict__ B,
                                                float* __restrict__ C,
                                                int N) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.f;
        } else {
            float sum = 0.f;
            // Only indices from col to row contribute, exploiting the sparsity of upper parts.
            for (int k = col; k <= row; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

// C++ interface exposed to PyTorch. It verifies tensor properties, and sets up the grid and block
// dimensions according to the configurable TILE_SIZE.

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Configure block and grid dimensions based on the adjustable TILE_SIZE.
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the CUDA kernel with the configured parameters.
    triangular_mm_kernel_optimized<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized triangular matrix multiplication (CUDA) with configurable block sizes");
}
