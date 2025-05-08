#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Experiment with different block sizes by defining TILE_SIZE at compile time.
// Recommended values: 16, 32 (which gives blockDim.x = blockDim.y = TILE_SIZE).
#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

// CUDA kernel: computes C = tril(A * B) for lower triangular matrices A and B.
// It uses shared memory tiling with a configurable TILE_SIZE to experiment with optimal block sizes
// on different hardware (e.g., NVIDIA H100). Only the valid portion (row >= col) is computed.
__global__ void triangular_mm_config_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int N) {
    // Allocate shared memory arrays with padding to avoid bank conflicts
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    // Determine global row and column indices
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Out-of-bound check
    if (row >= N || col >= N) return;

    // For upper triangular part, result is 0
    if (row < col) {
        C[row * N + col] = 0.f;
        return;
    }

    float sum = 0.f;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Loop over tiles
    for (int t = 0; t < numTiles; t++) {
        int tileStart = t * TILE_SIZE;

        // Load a tile of A from global memory into shared memory
        int a_col = tileStart + threadIdx.x;
        if (a_col < N && a_col <= row)
            As[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.f;

        // Load a tile of B from global memory into shared memory
        int b_row = tileStart + threadIdx.y;
        if (b_row < N && b_row >= col)
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.f;

        __syncthreads();

        // Determine the range of k to accumulate within this tile
        int k_start = max(tileStart, col);
        int k_end = min(tileStart + TILE_SIZE, row + 1);
        for (int k = k_start; k < k_end; k++) {
            int localK = k - tileStart;
            sum += As[threadIdx.y][localK] * Bs[localK][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}

// PyTorch interface function
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have the same dimensions");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Configure launch parameters based on TILE_SIZE
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_config_kernel<<<grid, block>>>(A.data_ptr<float>(),
                                                 B.data_ptr<float>(),
                                                 C.data_ptr<float>(),
                                                 N);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Configurable block size lower triangular matrix multiplication (CUDA)");
}
