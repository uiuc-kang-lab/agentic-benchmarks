#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <iostream>

// Define block and tile dimensions
#define BLOCK_SIZE 16
#define TILE_DIM (BLOCK_SIZE * 2)  // 32

// CUDA kernel combining register tiling with warp-aligned shared memory loading
__global__ void hybrid_matmul_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int K, int N) {
    // Compute starting indices for the tile
    int blockRow = blockIdx.y * TILE_DIM;
    int blockCol = blockIdx.x * TILE_DIM;

    // Each thread computes a 2x2 block
    int row = blockRow + threadIdx.y;               // first row this thread writes
    int col = blockCol + threadIdx.x;               // first column this thread writes
    int row2 = row + BLOCK_SIZE;                    // second row
    int col2 = col + BLOCK_SIZE;                    // second column

    // Registers to accumulate the 2x2 output
    float Cvalue00 = 0.0f, Cvalue01 = 0.0f, Cvalue10 = 0.0f, Cvalue11 = 0.0f;

    // Shared memory tiles for A and B
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    // Loop over tiles along the K dimension
    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < numTiles; t++) {
        // Global column index for A tile
        int Acol = t * TILE_DIM + threadIdx.x;
        // Global row index for B tile
        int Brow = t * TILE_DIM + threadIdx.y;

        // Load elements of A into shared memory for two rows per thread
        if ((row < M) && (Acol < K)) {
            As[threadIdx.y][threadIdx.x] = A[row * K + Acol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if ((row2 < M) && (Acol < K)) {
            As[threadIdx.y + BLOCK_SIZE][threadIdx.x] = A[row2 * K + Acol];
        } else {
            As[threadIdx.y + BLOCK_SIZE][threadIdx.x] = 0.0f;
        }

        // Load elements of B into shared memory for two columns per thread
        if ((Brow < K) && (col < N)) {
            Bs[threadIdx.y][threadIdx.x] = B[Brow * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if ((Brow < K) && (col2 < N)) {
            Bs[threadIdx.y][threadIdx.x + BLOCK_SIZE] = B[Brow * N + col2];
        } else {
            Bs[threadIdx.y][threadIdx.x + BLOCK_SIZE] = 0.0f;
        }

        __syncthreads();

        // Compute the partial results for this tile
        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            float a0 = As[threadIdx.y][k];                         // for first row
            float a1 = As[threadIdx.y + BLOCK_SIZE][k];              // for second row
            float b0 = Bs[k][threadIdx.x];                           // for first col
            float b1 = Bs[k][threadIdx.x + BLOCK_SIZE];              // for second col

            Cvalue00 += a0 * b0;
            Cvalue01 += a0 * b1;
            Cvalue10 += a1 * b0;
            Cvalue11 += a1 * b1;
        }

        __syncthreads();
    }

    // Write the computed sub-block to global memory
    if (row < M && col < N)
        C[row * N + col] = Cvalue00;
    if (row < M && col2 < N)
        C[row * N + col2] = Cvalue01;
    if (row2 < M && col < N)
        C[row2 * N + col] = Cvalue10;
    if (row2 < M && col2 < N)
        C[row2 * N + col2] = Cvalue11;
}

// Host function to launch the kernel or fallback to cuBLAS for larger matrices
void matmul_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // For large matrices, use cuBLAS to leverage highly optimized routines
    if (M > 512 || N > 512 || K > 512) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    B.data_ptr<float>(), N,
                    A.data_ptr<float>(), K,
                    &beta,
                    C.data_ptr<float>(), N);
        cublasDestroy(handle);
    } else {
        // Each block computes a TILE_DIM x TILE_DIM tile of the output
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

        hybrid_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(),
                                              B.data_ptr<float>(),
                                              C.data_ptr<float>(),
                                              M, K, N);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();
    }
}

// Pybind11 module registration
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", [](torch::Tensor A, torch::Tensor B) {
        TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
        TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
        TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
        TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

        auto C = torch::zeros({A.size(0), B.size(1)}, A.options());
        matmul_cuda(A, B, C);
        return C;
    }, "Hybrid register-tiling and warp-aligned matrix multiplication (CUDA)");
}
