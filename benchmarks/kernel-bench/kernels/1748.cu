#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// This kernel computes lower triangular matrix multiplication: C[r, c] = sum_{k=c}^{r} A[r, k] * B[k, c]
// It uses tiling with shared memory. The loop over tiles runs from tile index = blockIdx.x (for col) to blockIdx.y (for row).
// To optimize performance, __syncthreads() is called only when necessary: once after loading each tile and then only if more tiles remain.

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int N) {
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Early exit for blocks in the upper triangular region
    if (blockIdx.x > blockIdx.y) {
        if (row < N && col < N) {
            C[row * N + col] = 0.0f;
        }
        return;
    }

    // Only compute if within matrix bounds
    if (row < N && col < N) {
        // Only process lower triangular part (r >= c)
        if (row >= col) {
            // Determine the range of tile indices for the summation: from tile corresponding to col to tile corresponding to row
            int tile_start = blockIdx.x;  // because col = blockIdx.x * TILE_SIZE + threadIdx.x
            int tile_end = blockIdx.y;    // because row = blockIdx.y * TILE_SIZE + threadIdx.y
            
            for (int t = tile_start; t <= tile_end; t++) {
                // Load element for A: A[row, t*TILE_SIZE + threadIdx.x] if valid and if within lower triangular condition (<= row)
                int A_col = t * TILE_SIZE + threadIdx.x;
                if (A_col < N && A_col <= row) {
                    A_tile[threadIdx.y][threadIdx.x] = A[row * N + A_col];
                } else {
                    A_tile[threadIdx.y][threadIdx.x] = 0.0f;
                }

                // Load element for B: B[t*TILE_SIZE + threadIdx.y, col] if valid and if col is within lower triangular condition (col <= t*TILE_SIZE + threadIdx.y)
                int B_row = t * TILE_SIZE + threadIdx.y;
                if (B_row < N && col <= B_row) {
                    B_tile[threadIdx.y][threadIdx.x] = B[B_row * N + col];
                } else {
                    B_tile[threadIdx.y][threadIdx.x] = 0.0f;
                }

                // Synchronize to ensure tile is loaded before computation
                __syncthreads();

                // Compute partial sum from this tile. Each thread accumulates product for indices k in [0, TILE_SIZE).
                for (int k = 0; k < TILE_SIZE; k++) {
                    int global_k = t * TILE_SIZE + k;
                    if (global_k >= col && global_k <= row) {
                        sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
                    }
                }

                // For all but the last tile, synchronize so that the next iteration's loads do not conflict with ongoing reads
                if (t < tile_end) {
                    __syncthreads();
                }
            }
            C[row * N + col] = sum;
        } else {
            // In the upper triangular part, set output to zero
            C[row * N + col] = 0.0f;
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

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Triangular matrix multiplication with optimized synchronizations (CUDA)");
}
