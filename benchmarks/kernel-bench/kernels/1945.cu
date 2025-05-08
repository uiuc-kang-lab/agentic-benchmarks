/*
Combined CUDA kernel for lower-triangular matrix multiplication with shared memory tiling
and early block-level exit for blocks entirely in the upper triangular region.

This kernel computes C = tril(A * B) where A and B are lower-triangular matrices.
For each element (row, col) with row >= col, it computes:
   C[row, col] = sum_{k=col}^{row} A[row, k] * B[k, col]
Elements in the upper triangular region (row < col) are set to 0.

The kernel uses tile-based shared memory to reduce global memory accesses and
includes bounds- and triangular-structure checks to avoid unnecessary computations.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile size for shared memory tiling.
#define TILE_SIZE 32

// Optimized kernel using shared memory tiling and block-level pruning.
__global__ void optimized_shared_triangular_mm_kernel(const float* __restrict__ A,
                                                        const float* __restrict__ B,
                                                        float* __restrict__ C,
                                                        int N) {
    // Determine global row and column indices
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Skip threads that are out-of-bound
    if (row >= N || col >= N) return;

    // Early exit: if the entire block is in the upper triangular region
    // For the block, the maximum row index is: block_row_max = blockIdx.y * TILE_SIZE + TILE_SIZE - 1
    // and the minimum column index is: block_col_min = blockIdx.x * TILE_SIZE
    // If block_row_max < block_col_min, then for all (r, c) in the block, r < c.
    int block_row_max = blockIdx.y * TILE_SIZE + TILE_SIZE - 1;
    int block_col_min = blockIdx.x * TILE_SIZE;
    if (block_row_max < block_col_min) {
        C[row * N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;

    // Shared memory tiles for A and B
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    // Compute the number of tiles along the k-dimension
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Loop over tiles
    for (int m = 0; m < numTiles; m++) {
        // Global index for the k dimension for loading A and B
        int k_A = m * TILE_SIZE + threadIdx.x; // column index for A
        int k_B = m * TILE_SIZE + threadIdx.y; // row index for B

        // Load A[row, k] into shared memory if within bounds and valid (nonzero only if k <= row)
        if (k_A < N && row >= k_A) {
            sA[threadIdx.y][threadIdx.x] = A[row * N + k_A];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B[k, col] into shared memory if within bounds and valid (nonzero only if k >= col)
        if (k_B < N && k_B >= col) {
            sB[threadIdx.y][threadIdx.x] = B[k_B * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize to ensure the tile is fully loaded
        __syncthreads();

        // Define the global k range for this tile
        int tile_start = m * TILE_SIZE;
        int tile_end = tile_start + TILE_SIZE;
        if (tile_end > N) tile_end = N;

        // For a valid multiplication for lower triangular matrices, the summation index k must satisfy:
        //    col <= k <= row, and also k is within [tile_start, tile_end).
        int k_lower = (col > tile_start) ? col : tile_start;      // max(col, tile_start)
        int k_upper = (row < (tile_end - 1)) ? row : (tile_end - 1); // min(row, tile_end-1)

        // Convert these global indices to local tile indices
        int local_start = (k_lower - tile_start);
        int local_end = (k_upper - tile_start + 1);

        // Accumulate sum if there is a valid range
        for (int k = local_start; k < local_end; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // If current element is in the upper triangular portion (row < col), enforce zero.
    if (row < col) {
        sum = 0.0f;
    }
    
    C[row * N + col] = sum;
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

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    optimized_shared_triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Optimized Shared Memory Triangular Matrix Multiplication (CUDA)");
}
