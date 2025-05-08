#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_SIZE 32

// Device function to load a tile from matrix A with transposed access.
// A: shape (K, M), accessed as A[k * M + row] for A^T
// row: corresponds to the output row index (and A's column index)
// t: current tile index along the K dimension
__device__ inline void load_tile_a(const float* __restrict__ A, float tileA[TILE_SIZE][TILE_SIZE], int t, int M, int K, int row) {
    int global_k = t * TILE_SIZE + threadIdx.x;
    if (row < M && global_k < K)
        tileA[threadIdx.y][threadIdx.x] = A[global_k * M + row];
    else
        tileA[threadIdx.y][threadIdx.x] = 0.0f;
}

// Device function to load a tile from matrix B.
// B: shape (K, N), accessed as B[k * N + col]
// col: corresponds to the output column index
// t: current tile index along the K dimension
__device__ inline void load_tile_b(const float* __restrict__ B, float tileB[TILE_SIZE][TILE_SIZE], int t, int N, int K, int col) {
    int global_k = t * TILE_SIZE + threadIdx.y;
    if (global_k < K && col < N)
        tileB[threadIdx.y][threadIdx.x] = B[global_k * N + col];
    else
        tileB[threadIdx.y][threadIdx.x] = 0.0f;
}

// Device function to compute the dot product for the current tile.
// Performs the fused multiply-add over the tile's k-dimension.
__device__ inline float compute_partial_sum(const float tileA[TILE_SIZE][TILE_SIZE],
                                              const float tileB[TILE_SIZE][TILE_SIZE]) {
    float partial = 0.0f;
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
        partial = __fmaf_rn(tileA[threadIdx.y][k], tileB[k][threadIdx.x], partial);
    }
    return partial;
}

// Kernel to compute C = A^T * B using modular device functions for tile loading and computation.
// A: shape (K, M), B: shape (K, N), C: shape (M, N)
// Each thread computes one element of C as sum_{k=0}^{K-1} A[k, i] * B[k, j], with A accessed transposed.
__global__ void modularDeviceFunctionsTilingKernel(const float* __restrict__ A,
                                                    const float* __restrict__ B,
                                                    float* __restrict__ C,
                                                    int K, int M, int N) {
    // Compute global output indices
    int row = blockIdx.x * TILE_SIZE + threadIdx.y; // corresponds to A's column index (i in C)
    int col = blockIdx.y * TILE_SIZE + threadIdx.x;     // corresponds to B's column index (j in C)
    
    float sum = 0.0f;

    // Allocate shared memory tiles for A and B
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        // Load tile for A with transposed access and tile for B
        load_tile_a(A, tileA, t, M, K, row);
        load_tile_b(B, tileB, t, N, K, col);

        __syncthreads();

        // Compute the partial dot product for this tile
        sum += compute_partial_sum(tileA, tileB);

        __syncthreads();
    }

    // Write the computed value to C if within valid indices
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// The forward function exposed via PyBind11.
// A: Tensor of shape (K, M) [CUDA, float32]
// B: Tensor of shape (K, N) [CUDA, float32]
// Returns: Tensor C of shape (M, N) computed as A^T * B
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    // Allocate output tensor C of shape (M, N)
    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Define thread block and grid sizes
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((M + TILE_SIZE - 1) / TILE_SIZE,
                 (N + TILE_SIZE - 1) / TILE_SIZE);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    modularDeviceFunctionsTilingKernel<<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B (CUDA) using modular device functions for tiling");
}
