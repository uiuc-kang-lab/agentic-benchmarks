/*
 * This CUDA kernel computes C = A.T * B where A is of shape (K, M) and B is of shape (K, N).
 * It combines tiling and shared memory loading (from Kernel 1) with a non-atomic grid mapping (from Kernel 2).
 * Each block computes a TILE_M x TILE_N tile of C by looping over K in chunks of BLOCK_K, loading subtiles of A and B
 * into shared memory to improve global memory reuse and minimize divergence.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Tile dimensions
#define TILE_M 16  // Tile size in the M dimension (output row)
#define TILE_N 16  // Tile size in the N dimension (output column)
#define BLOCK_K 32 // Chunk size along the K dimension

__global__ void tiledSharedKernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int K,
                                    int M,
                                    int N) {
    // Each block computes a tile of output C of size TILE_M x TILE_N
    int row = blockIdx.x * TILE_M + threadIdx.x; // Corresponds to output index i
    int col = blockIdx.y * TILE_N + threadIdx.y; // Corresponds to output index j

    float value = 0.0f;

    // Allocate shared memory for a tile of A and B for the current k-chunk
    // A is accessed as A[k, i] where i corresponds to row index in C
    // B is accessed as B[k, j]
    __shared__ float As[BLOCK_K][TILE_M];
    __shared__ float Bs[BLOCK_K][TILE_N];

    // Loop over K in chunks of size BLOCK_K
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // Use all threads in the block to load the sub-tiles into shared memory.
        // There are TILE_M * TILE_N threads per block but we need to load BLOCK_K * TILE_M elements for A and BLOCK_K * TILE_N for B.
        int tid = threadIdx.y * TILE_M + threadIdx.x;  // Unique thread id in the block
        int numThreads = TILE_M * TILE_N;  // Total threads per block

        // Load a tile from A into shared memory.
        // A is stored in global memory as A[k * M + i] where i indexes the M dimension.
        for (int idx = tid; idx < BLOCK_K * TILE_M; idx += numThreads) {
            int t = idx / TILE_M;      // iterate over k dimension in the tile
            int m_idx = idx % TILE_M;    // iterate over the tile row (which corresponds to global row index in C)
            int global_i = blockIdx.x * TILE_M + m_idx;
            int global_k = k0 + t;
            if (global_i < M && global_k < K)
                As[t][m_idx] = A[global_k * M + global_i];
            else
                As[t][m_idx] = 0.0f;
        }

        // Load a tile from B into shared memory.
        // B is stored as B[k * N + j] where j indexes the N dimension.
        for (int idx = tid; idx < BLOCK_K * TILE_N; idx += numThreads) {
            int t = idx / TILE_N;      // iterate over k dimension in the tile
            int n_idx = idx % TILE_N;    // iterate over the tile column (which corresponds to global column index in C)
            int global_j = blockIdx.y * TILE_N + n_idx;
            int global_k = k0 + t;
            if (global_j < N && global_k < K)
                Bs[t][n_idx] = B[global_k * N + global_j];
            else
                Bs[t][n_idx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product for this k-chunk
        // Each thread accumulates the product of the corresponding A and B tile elements
        for (int t = 0; t < BLOCK_K; ++t) {
            value += As[t][threadIdx.x] * Bs[t][threadIdx.y];
        }

        __syncthreads();
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

    tiledSharedKernel<<<grid, block>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using tiled shared memory (CUDA)");
}
