#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Define tile size for shared memory tiling
#define TILE_SIZE 16

// Device function to load a tile of A in transposed order from global memory into shared memory.
// A is of shape (K, M) stored in row-major order, where element A[k, i] is at A[k * M + i].
// The tile of A is loaded such that tileA[ty][tx] corresponds to A[tile*TILE_SIZE + tx, row].
__device__ void loadATile(const float* __restrict__ A, float tileA[TILE_SIZE][TILE_SIZE], 
                           int tile, int row, int M, int K) {
    int col = tile * TILE_SIZE + threadIdx.x;
    if (row < M && col < K) {
        // Load A in transposed order: C[i,j] = sum_k A[k,i] * B[k,j]
        tileA[threadIdx.y][threadIdx.x] = A[col * M + row];
    } else {
        tileA[threadIdx.y][threadIdx.x] = 0.0f;
    }
}

// Device function to load a tile of B from global memory into shared memory.
// B is of shape (K, N) where element B[k, j] is at B[k * N + j].
__device__ void loadBTile(const float* __restrict__ B, float tileB[TILE_SIZE][TILE_SIZE], 
                           int tile, int col, int N, int K) {
    int row = tile * TILE_SIZE + threadIdx.y;
    if (row < K && col < N) {
        tileB[threadIdx.y][threadIdx.x] = B[row * N + col];
    } else {
        tileB[threadIdx.y][threadIdx.x] = 0.0f;
    }
}

// Device function to compute the dot product for the current tile loaded in shared memory.
__device__ float computeTileProduct(const float tileA[TILE_SIZE][TILE_SIZE], 
                                      const float tileB[TILE_SIZE][TILE_SIZE]) {
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
        sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }
    return sum;
}

// CUDA kernel that computes C = A.T * B using modular device functions.
// A: shape (K, M), B: shape (K, N), C: shape (M, N).
// Each thread computes one element of C: C[i,j] = sum_over_k A[k,i] * B[k,j].
__global__ void matMulModularKernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int K, int M, int N) {
    // Compute output indices for C (row corresponds to A's column index i, col corresponds to B's column index j)
    int row = blockIdx.x * TILE_SIZE + threadIdx.y; // i
    int col = blockIdx.y * TILE_SIZE + threadIdx.x; // j

    float sum = 0.0f;

    // Declare shared memory tiles for A and B
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        // Load tile of A (transposed load) and B using defined device functions
        loadATile(A, tileA, t, row, M, K);
        loadBTile(B, tileB, t, col, N, K);

        __syncthreads(); // Ensure the tiles are loaded

        // Compute partial product from the loaded tile
        sum += computeTileProduct(tileA, tileB);

        __syncthreads(); // Synchronize before loading the next tile
    }

    // Write result to C if within bounds
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Forward function exposed via PyBind11
// Performs the operation C = A.T * B where A is of shape (K, M) and B is of shape (K, N).
// C will have shape (M, N).

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Define thread block and grid sizes
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Launch the kernel
    matMulModularKernel<<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B (CUDA) using modular device functions");
}
