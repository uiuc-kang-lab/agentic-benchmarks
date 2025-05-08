#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile size for shared memory tiling
#define TILE_SIZE 16

// CUDA kernel for matrix multiplication with transposed inputs using shared memory tiling
// C = A.T * B.T  where A is (K x M) and B is (N x K), so C is (M x N) computed as:
//   C[m, n] = sum_{k=0}^{K-1} A[k, m] * B[n, k]
// The kernel loads tiles of A and B into shared memory, and conditionally synchronizes threads
// using __syncthreads() only when necessary to ensure shared memory consistency. The final __syncthreads()
// at the end of the loop is skipped to reduce overhead.

template <typename scalar_t>
__global__ void matmul_transpose_tiled_kernel(
    const scalar_t* __restrict__ A,  // A: (K x M), stored as A[k*M + m]
    const scalar_t* __restrict__ B,  // B: (N x K), stored as B[n*K + k]
    scalar_t* __restrict__ C,        // C: (M x N), stored as C[m*N + n]
    const int M,                     // Number of rows in C and A's second dim
    const int N,                     // Number of columns in C and B's first dim
    const int K) {                   // Summation dimension

  // Compute thread indices within block
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Compute global indices for output C
  int m = blockIdx.y * TILE_SIZE + ty;  // row index in C (and A's 2nd dim)
  int n = blockIdx.x * TILE_SIZE + tx;  // column index in C (and B's 1st dim)

  scalar_t sum = 0;

  // Shared memory tiles for A and B
  __shared__ scalar_t A_tile[TILE_SIZE][TILE_SIZE];
  __shared__ scalar_t B_tile[TILE_SIZE][TILE_SIZE];

  // Number of tiles over the K dimension
  int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (int t = 0; t < num_tiles; t++) {
    // Load a tile of A from global memory into shared memory
    // A is stored as (K x M): element A[k, m] = A[k * M + m]
    int k_index_A = t * TILE_SIZE + tx;  // k index for A load
    if (m < M && k_index_A < K) {
      A_tile[ty][tx] = A[k_index_A * M + m];
    } else {
      A_tile[ty][tx] = 0;
    }

    // Load a tile of B from global memory into shared memory
    // B is stored as (N x K): element B[n, k] = B[n * K + k]
    int k_index_B = t * TILE_SIZE + ty;  // k index for B load
    if (n < N && k_index_B < K) {
      B_tile[ty][tx] = B[n * K + k_index_B];
    } else {
      B_tile[ty][tx] = 0;
    }

    // Synchronize to ensure the tiles are loaded before computation
    __syncthreads();

    // Compute partial product for this tile
    #pragma unroll
    for (int i = 0; i < TILE_SIZE; i++) {
      sum += A_tile[ty][i] * B_tile[i][tx];
    }

    // Synchronize threads before loading the next tile, except for the last iteration
    if (t < num_tiles - 1) {
      __syncthreads();
    }
  }

  // Write the result to global memory
  if (m < M && n < N) {
    C[m * N + n] = sum;
  }
}

// PyTorch interface: wraps the CUDA kernel into a function callable from Python
torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
  // Dimensions: A is (K x M), B is (N x K) so that C is (M x N)
  const int K = A.size(0);
  const int M = A.size(1);
  const int N = B.size(0);

  // Allocate output tensor
  auto C = torch::empty({M, N}, A.options());

  // Define grid and block dimensions
  dim3 threads(TILE_SIZE, TILE_SIZE);
  dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

  AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_transpose_tiled_kernel", ([&] {
    matmul_transpose_tiled_kernel<scalar_t><<<blocks, threads>>>(
      A.data_ptr<scalar_t>(),
      B.data_ptr<scalar_t>(),
      C.data_ptr<scalar_t>(),
      M, N, K);
  }));

  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &matmul_transpose_cuda, "Matrix multiplication with transposed inputs using tiled shared-memory kernel (CUDA)");
}
