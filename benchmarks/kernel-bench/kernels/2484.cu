#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define the tile size; can be tuned (e.g., 16, 32, etc.) for optimal performance
#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

// Inline function to load a tile from matrix A (stored in transposed form)
// A is of shape (K, M) and element A(k, row) is accessed as A[k * M + row]
template <typename scalar_t>
__device__ __forceinline__ void load_tile_A(
    scalar_t (&tileA)[TILE_SIZE][TILE_SIZE],
    const scalar_t* __restrict__ A,
    const int row,
    const int tile_idx,
    const int M,
    const int K) {
  int k_index = tile_idx * TILE_SIZE + threadIdx.y;
  if (k_index < K && row < M)
    tileA[threadIdx.y][threadIdx.x] = A[k_index * M + row];
  else
    tileA[threadIdx.y][threadIdx.x] = static_cast<scalar_t>(0);
}

// Inline function to load a tile from matrix B (stored in transposed form)
// B is of shape (N, K) and element B(col, k) is accessed as B[col * K + k]
template <typename scalar_t>
__device__ __forceinline__ void load_tile_B(
    scalar_t (&tileB)[TILE_SIZE][TILE_SIZE],
    const scalar_t* __restrict__ B,
    const int col,
    const int tile_idx,
    const int N,
    const int K) {
  int k_index = tile_idx * TILE_SIZE + threadIdx.x;
  if (k_index < K && col < N)
    tileB[threadIdx.y][threadIdx.x] = B[col * K + k_index];
  else
    tileB[threadIdx.y][threadIdx.x] = static_cast<scalar_t>(0);
}

// Compute the partial sum for one tile using loop unrolling
template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_tile_sum(
    const scalar_t (&tileA)[TILE_SIZE][TILE_SIZE],
    const scalar_t (&tileB)[TILE_SIZE][TILE_SIZE]) {
  scalar_t sum = 0;
  // Use two-way unrolling to better utilize instruction-level parallelism
  // while maintaining register pressure
  #pragma unroll 2
  for (int k = 0; k < TILE_SIZE; k += 2) {
    sum = fma(tileA[k][threadIdx.x], tileB[threadIdx.y][k], sum);
    if (k + 1 < TILE_SIZE) {
      sum = fma(tileA[k+1][threadIdx.x], tileB[threadIdx.y][k+1], sum);
    }
  }
  return sum;
}

// Combined CUDA kernel for performing matrix multiplication with transposed inputs
// Computes C = A^T * B^T where A is (K, M) and B is (N, K), resulting in C of shape (M, N)
template <typename scalar_t>
__global__ void combined_matmul_transpose_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {

  // Compute the global row and column indices for C
  const int row = blockIdx.x * TILE_SIZE + threadIdx.x;
  const int col = blockIdx.y * TILE_SIZE + threadIdx.y;

  __shared__ scalar_t tileA[TILE_SIZE][TILE_SIZE];
  __shared__ scalar_t tileB[TILE_SIZE][TILE_SIZE];

  scalar_t sum = 0;
  const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

  // Loop over all tiles
  for (int t = 0; t < num_tiles; ++t) {
    load_tile_A(tileA, A, row, t, M, K);
    load_tile_B(tileB, B, col, t, N, K);

    __syncthreads();

    // Accumulate the product from the current tile
    sum += compute_tile_sum(tileA, tileB);

    __syncthreads();
  }

  // Write the result to C if within bounds
  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

// Entry point for the CUDA extension
// A: Tensor of shape (K, M) in transposed layout
// B: Tensor of shape (N, K) in transposed layout
// C will be computed as C = A^T * B^T, resulting in (M, N)

torch::Tensor combined_matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
  const int K = A.size(0);
  const int M = A.size(1);
  const int N = B.size(0);

  auto C = torch::empty({M, N}, A.options());

  dim3 threads(TILE_SIZE, TILE_SIZE);
  dim3 blocks((M + TILE_SIZE - 1) / TILE_SIZE,
              (N + TILE_SIZE - 1) / TILE_SIZE);

  AT_DISPATCH_FLOATING_TYPES(A.type(), "combined_matmul_transpose_kernel", ([&] {
    combined_matmul_transpose_kernel<scalar_t><<<blocks, threads>>>(
        A.data_ptr<scalar_t>(),
        B.data_ptr<scalar_t>(),
        C.data_ptr<scalar_t>(),
        M, N, K);
  }));

  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &combined_matmul_transpose_cuda, "Combined Efficient Matrix Multiplication with Transposed Inputs (CUDA)");
}
