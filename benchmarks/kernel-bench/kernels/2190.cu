#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Tile dimensions
#define TILE_M 16  // Tile size in the M dimension (output row)
#define TILE_N 16  // Tile size in the N dimension (output column)
#define BLOCK_K 32 // Chunk size along the K dimension

// This kernel uses warp-level primitives to reduce shared memory usage and improve performance.
__global__ void warpOptimizedKernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int K,
                                    int M,
                                    int N) {
    int row = blockIdx.x * TILE_M + threadIdx.x; // Output row
    int col = blockIdx.y * TILE_N + threadIdx.y; // Output column

    float value = 0.0f;

    // Loop over K in chunks of size BLOCK_K
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // Load sub-tiles of A and B into registers
        float a_reg[TILE_M];
        float b_reg[TILE_N];

        // Use warp shuffle to load data into registers
        for (int t = 0; t < BLOCK_K; ++t) {
            int global_i = row;
            int global_j = col;
            int global_k = k0 + t;

            if (global_i < M && global_k < K) {
                a_reg[threadIdx.x] = A[global_k * M + global_i];
            } else {
                a_reg[threadIdx.x] = 0.0f;
            }

            if (global_j < N && global_k < K) {
                b_reg[threadIdx.y] = B[global_k * N + global_j];
            } else {
                b_reg[threadIdx.y] = 0.0f;
            }

            // Perform warp-level reduction
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                a_reg[threadIdx.x] += __shfl_down_sync(0xFFFFFFFF, a_reg[threadIdx.x], offset);
                b_reg[threadIdx.y] += __shfl_down_sync(0xFFFFFFFF, b_reg[threadIdx.y], offset);
            }

            // Accumulate results
            value += a_reg[threadIdx.x] * b_reg[threadIdx.y];
        }
    }

    // Write the computed value to the output matrix C if within bounds
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

// Forward function exposed to PyTorch
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    // Dimensions: A: (K, M) and B: (K, N). We compute C = A.T * B, so C has shape (M, N).
    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Define block dimensions: each block computes a TILE_M x TILE_N tile of C
    dim3 block(TILE_M, TILE_N);
    // Grid covers the M and N dimensions
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    warpOptimizedKernel<<<grid, block>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using warp-level optimization (CUDA)");
}
