#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

// Define block dimensions and tile size for the inner (K) dimension
#define BLOCK_ROWS 16
#define BLOCK_COLS 32  // 16 x 32 = 512 threads per block
#define TILE_K 16

// Kernel: Tiled matrix multiplication for 3D tensor-matrix multiplication
// We flatten the first two dimensions (N and M) of A (shape: N x M x K) to a 2D matrix of shape (N*M x K).
// B is of shape (K x L) and the output is (N*M x L), which is interpreted as (N x M x L).

template <typename scalar_t>
__global__ void tiled_blocksize_512_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int num_rows,  // equals N*M
    int K,
    int L) {

    // Compute global row and column indices for the output matrix
    int row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
    int col = blockIdx.x * BLOCK_COLS + threadIdx.x;
    
    scalar_t sum = 0;

    // Shared memory tiles
    __shared__ scalar_t s_A[BLOCK_ROWS][TILE_K]; // Tile from A: size BLOCK_ROWS x TILE_K
    __shared__ scalar_t s_B[TILE_K][BLOCK_COLS];   // Tile from B: size TILE_K x BLOCK_COLS

    // Loop over tiles along the K dimension
    int numTiles = (K + TILE_K - 1) / TILE_K;
    for (int t = 0; t < numTiles; t++) {
        // Load tile from A into shared memory
        int aCol = t * TILE_K + threadIdx.x;  // Only threads with threadIdx.x < TILE_K load valid data
        if (threadIdx.x < TILE_K && row < num_rows && aCol < K) {
            s_A[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        } else if (threadIdx.x < TILE_K) {
            s_A[threadIdx.y][threadIdx.x] = 0;
        }

        // Load tile from B into shared memory
        int bRow = t * TILE_K + threadIdx.y;  // All threads in y dimension [0, BLOCK_ROWS) load B if within limit
        if (threadIdx.y < TILE_K && bRow < K && col < L) {
            s_B[threadIdx.y][threadIdx.x] = B[bRow * L + col];
        } else if (threadIdx.y < TILE_K) {
            s_B[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Perform multiplication on the tile
        for (int i = 0; i < TILE_K; i++) {
            sum += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result to global memory
    if (row < num_rows && col < L) {
        output[row * L + col] = sum;
    }
}

// CUDA forward function
void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);
    int num_rows = N * M; // Flatten N and M

    // Set up grid and block dimensions
    dim3 block(BLOCK_COLS, BLOCK_ROWS); // 32 x 16 = 512 threads per block
    dim3 grid((L + BLOCK_COLS - 1) / BLOCK_COLS, (num_rows + BLOCK_ROWS - 1) / BLOCK_ROWS);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "tiled_blocksize_512_cuda_forward", ([&] {
      tiled_blocksize_512_kernel<scalar_t><<<grid, block>>>(
          A.data_ptr<scalar_t>(),
          B.data_ptr<scalar_t>(),
          output.data_ptr<scalar_t>(),
          num_rows, K, L);
    }));

    // Check for any errors launching the kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in module_fn_cuda_forward: %s\n", cudaGetErrorString(err));
    }
}

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor module_fn_forward(
    torch::Tensor A,
    torch::Tensor B) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  
  int N = A.size(0);
  int M = A.size(1);
  int L = B.size(1);

  // Allocate output tensor with shape (N, M, L). Memory layout is contiguous so it can be treated as (N*M x L).
  auto output = torch::zeros({N, M, L}, A.options());
  module_fn_cuda_forward(A, B, output);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &module_fn_forward, "Tiled 3D tensor-matrix multiplication with block size 512 (CUDA)");
}
