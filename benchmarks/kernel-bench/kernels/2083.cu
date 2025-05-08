/*
 * This CUDA extension implements a fused triangular matrix multiplication kernel.
 * It combines the lower-triangular dot product computation and the zeroing of the
 * upper triangular part into a single kernel. Using a 2D grid (with grid-stride loops),
 * each thread computes one element of the output matrix C = A * B for square, lower-
 * triangular matrices A and B. This avoids the overhead of mapping 1D indices (with sqrt)
 * as in kernel2 and the stream synchronization overhead of kernel1. 
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

// Fused kernel: each thread computes one C[row, col]. 
// For a lower triangular element (row >= col), it computes the dot product sum_{k=col}^{row} A[row, k] * B[k, col].
// For an upper triangular element (row < col), it simply writes 0. 
// The kernel uses grid-stride loops for robustness when N is large.

__global__ void fused_triangular_mm_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int N) {
    // 2D thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // use grid-stride loops to cover entire matrix
    for (int r = row; r < N; r += gridDim.y * blockDim.y) {
        for (int c = col; c < N; c += gridDim.x * blockDim.x) {
            if (r < c) {
                C[r * N + c] = 0.f;
            } else {
                float sum = 0.f;
                // Only iterate from 'c' to 'r' because matrices are lower triangular
                for (int k = c; k <= r; ++k) {
                    sum += A[r * N + k] * B[k * N + c];
                }
                C[r * N + c] = sum;
            }
        }
    }
}

// Forward function for PyTorch extension
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

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (N + TILE_SIZE - 1) / TILE_SIZE);

    fused_triangular_mm_kernel<<<grid, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Fused lower triangular matrix multiplication (CUDA)");
}
