#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// Kernel using shared memory for frequently reused data
__global__ void shared_memory_triangular_mm_kernel(const float* __restrict__ A,
                                                    const float* __restrict__ B,
                                                    float* __restrict__ C,
                                                    int N) {
    // Compute global row and column indices
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Only work on valid indices
    if (row < N && col < N) {
        // For upper-triangular region, set result to zero
        if (row < col) {
            C[row * N + col] = 0.f;
            return;
        }

        float sum = 0.f;

        // Shared memory tiles for A and B
        __shared__ float sA[TILE_SIZE][TILE_SIZE];
        __shared__ float sB[TILE_SIZE][TILE_SIZE];

        // Number of tiles along the k-dimension
        int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

        for (int m = 0; m < numTiles; m++) {
            // Calculate the global k-index for loading
            int col_idx_A = m * TILE_SIZE + threadIdx.x;  // For A(row, k)
            int row_idx_B = m * TILE_SIZE + threadIdx.y;    // For B(k, col)

            // Load elements into shared memory with bounds-checking
            sA[threadIdx.y][threadIdx.x] = (col_idx_A < N && row >= col_idx_A) ? A[row * N + col_idx_A] : 0.f;
            sB[threadIdx.y][threadIdx.x] = (row_idx_B < N) ? B[row_idx_B * N + col] : 0.f;

            // Synchronize to ensure the tile is loaded
            __syncthreads();

            // Determine the valid k-range within this tile.
            // The summation index (global k) should lie in [col, row] for lower triangular matrices.
            int global_k_start = m * TILE_SIZE;
            int local_start = (col > global_k_start) ? (col - global_k_start) : 0;
            int global_k_end = global_k_start + TILE_SIZE;
            int local_end = (row + 1 < global_k_end) ? (row + 1 - global_k_start) : TILE_SIZE;

            // Accumulate partial sum only for k in [local_start, local_end)
            for (int k = local_start; k < local_end; ++k) {
                sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
            }

            // Synchronize before loading the next tile to avoid race conditions.
            // This sync is omitted in the last iteration to avoid an unnecessary barrier.
            if (m != numTiles - 1)
                __syncthreads();
        }

        // Write the computed result back to global memory
        C[row * N + col] = sum;
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

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the optimized kernel
    shared_memory_triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Shared Memory Optimized Triangular Matrix Multiplication (CUDA)");
}