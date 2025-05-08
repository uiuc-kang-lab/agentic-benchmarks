#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_M 32
#define TILE_N 32
#define TILE_K 8

// This kernel computes C = A.T * B, treating A as the transpose of A (i.e. A'(i,k) = A[k*M + i]).
// A is of shape (K, M), B is of shape (K, N) and C is of shape (M, N), with row-major storage.
// The kernel uses shared memory tiling to load sub-blocks of A' and B to reduce global memory traffic.
// The work for loading tiles into shared memory is evenly distributed among threads using grid-stride loops.

__global__ void tiledMatmulKernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int K, int M, int N) {
    // Compute the global row and column indices for C
    int row = blockIdx.x * TILE_M + threadIdx.y;  // corresponds to row index in C and A'
    int col = blockIdx.y * TILE_N + threadIdx.x;  // corresponds to column index in C and B

    float sum = 0.0f;

    // Allocate shared memory for a tile of A' and a tile of B
    __shared__ float As[TILE_M][TILE_K];  // Tile for A' (A transposed view)
    __shared__ float Bs[TILE_K][TILE_N];    // Tile for B

    // Calculate the number of tiles needed along the K dimension
    int numTiles = (K + TILE_K - 1) / TILE_K;

    // Use a grid-stride loop to evenly distribute work for loading shared memory
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;

    for (int t = 0; t < numTiles; ++t) {
         // Load tile from A' into shared memory
         // A'(i, k) = A[k*M + i], so for the current tile:
         //   global row = blockIdx.x * TILE_M + local_row
         //   global k = t * TILE_K + local_col
         for (int index = tid; index < TILE_M * TILE_K; index += blockSize) {
              int local_row = index / TILE_K;
              int local_col = index % TILE_K;
              int global_row = blockIdx.x * TILE_M + local_row;
              int global_k = t * TILE_K + local_col;
              if (global_row < M && global_k < K)
                   As[local_row][local_col] = A[global_k * M + global_row];
              else
                   As[local_row][local_col] = 0.0f;
         }
         
         // Load tile from B into shared memory
         // B is of shape (K, N), accessed as B[k*N + j]
         // for the current tile:
         //   global k = t * TILE_K + local_row
         //   global col = blockIdx.y * TILE_N + local_col
         for (int index = tid; index < TILE_K * TILE_N; index += blockSize) {
              int local_row = index / TILE_N;
              int local_col = index % TILE_N;
              int global_k = t * TILE_K + local_row;
              int global_col = blockIdx.y * TILE_N + local_col;
              if (global_k < K && global_col < N)
                   Bs[local_row][local_col] = B[global_k * N + global_col];
              else
                   Bs[local_row][local_col] = 0.0f;
         }
         __syncthreads();

         // Compute partial dot product for this tile
         if (row < M && col < N) {
              for (int k = 0; k < TILE_K; ++k) {
                   sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
              }
         }
         __syncthreads();
    }

    // Write the final result to C
    if (row < M && col < N) {
         C[row * N + col] = sum;
    }
}

// The forward function, exposed via PyBind11, partitions the work according to the dimensions of C (M x N)
// and launches the tiledMatmulKernel to compute C = A.T * B.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have same first dimension (K)");
    int N = B.size(1);

    // Allocate the output tensor C of shape (M, N)
    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Define block and grid dimensions
    dim3 blockDim(TILE_N, TILE_M);  // blockDim.x = TILE_N, blockDim.y = TILE_M
    dim3 gridDim((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    tiledMatmulKernel<<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
         throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using shared memory tiling (CUDA)");
}
