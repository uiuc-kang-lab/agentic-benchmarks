#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_SIZE 32

// Device function to load a tile of matrix A (transposed access).
// A: shape (K, M) stored in row-major order, but we access it as A[k * M + row].
// row: global row index in C (i.e. column index in A).
// t: current tile index along the K dimension.
__device__ inline void loadTileA(const float* __restrict__ A, float tileA[TILE_SIZE][TILE_SIZE], int t, int M, int K, int row) {
    #pragma unroll
    int aIndex = t * TILE_SIZE + threadIdx.x;
    if (row < M && aIndex < K)
        tileA[threadIdx.y][threadIdx.x] = A[aIndex * M + row];
    else
        tileA[threadIdx.y][threadIdx.x] = 0.0f;
}

// Device function to load a tile of matrix B.
// B: shape (K, N) stored in row-major order, accessed as B[k * N + col].
// col: global column index in C (and B).
// t: current tile index along the K dimension.
__device__ inline void loadTileB(const float* __restrict__ B, float tileB[TILE_SIZE][TILE_SIZE], int t, int N, int K, int col) {
    int bIndex = t * TILE_SIZE + threadIdx.y;
    if (bIndex < K && col < N)
        tileB[threadIdx.y][threadIdx.x] = B[bIndex * N + col];
    else
        tileB[threadIdx.y][threadIdx.x] = 0.0f;
}

// Device function to compute the partial dot product from the loaded tiles.
// Each thread computes a dot product of one row of tileA and one column of tileB.
__device__ inline float computeTileSum(const float tileA[TILE_SIZE][TILE_SIZE],
                                         const float tileB[TILE_SIZE][TILE_SIZE]) {
    float sum_local = 0.0f;
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
        sum_local = __fmaf_rn(tileA[threadIdx.y][k], tileB[k][threadIdx.x], sum_local);
    }
    return sum_local;
}

// Modular kernel for computing C = A.T * B.
// A: shape (K, M), B: shape (K, N), C: shape (M, N).
// Note: A is accessed in a transposed manner: A.T(i, k) = A(k, i).
__global__ void matMulModularKernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int K, int M, int N) {
    // Compute global row and column indices for C.
    int row = blockIdx.x * TILE_SIZE + threadIdx.y; // corresponds to A's column index
    int col = blockIdx.y * TILE_SIZE + threadIdx.x; // corresponds to B's column index

    float sum = 0.0f;
    
    // Shared memory tiles for A and B
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Load a tile of A (with transposed access) and B into shared memory using helper functions.
        loadTileA(A, tileA, t, M, K, row);
        loadTileB(B, tileB, t, N, K, col);

        __syncthreads();

        // Compute partial dot product for this tile
        sum += computeTileSum(tileA, tileB);

        __syncthreads();
    }

    // Write the result back to C if within boundaries
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// The forward function exposed via PyBind11.
// Inputs:
//   A: Tensor of shape (K, M) [CUDA, float32]
//   B: Tensor of shape (K, N) [CUDA, float32]
// Returns:
//   C: Tensor of shape (M, N) computed as A.T * B.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    // Allocate output tensor C of shape (M, N).
    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Define thread block and grid sizes.
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Launch the modular tiled kernel.
    matMulModularKernel<<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B (CUDA) using modular device functions and shared memory tiling");
}
