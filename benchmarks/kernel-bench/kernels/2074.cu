#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

// CUDA kernel using grid-stride loops to handle workloads larger than the number of threads
__global__ void triangular_mm_kernel_strideloop(const float* __restrict__ A,
                                                  const float* __restrict__ B,
                                                  float* __restrict__ C,
                                                  int N) {
    // Starting indices for each thread
    int row_start = blockIdx.y * blockDim.y + threadIdx.y;
    int col_start = blockIdx.x * blockDim.x + threadIdx.x;

    // Stride sizes across the grid
    int row_stride = blockDim.y * gridDim.y;
    int col_stride = blockDim.x * gridDim.x;

    for (int row = row_start; row < N; row += row_stride) {
        for (int col = col_start; col < N; col += col_stride) {
            if (row < col) {
                C[row * N + col] = 0.f;
            } else {
                float sum = 0.f;
                // Only indices from col to row contribute in lower triangular multiplication
                for (int k = col; k <= row; ++k) {
                    sum += A[row * N + k] * B[k * N + col];
                }
                C[row * N + col] = sum;
            }
        }
    }
}

// C++ interface exposed to PyTorch
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

    // Configure block and grid dimensions; grid-stride loops will ensure full coverage
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_kernel_strideloop<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Grid-stride loop based triangular matrix multiplication (CUDA)");
}
