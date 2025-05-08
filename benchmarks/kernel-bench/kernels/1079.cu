#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

// BLOCK_SIZE can be tuned at compile time (e.g., 32, 64, 128, 256, 512)
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// Fix block rows to 16, then block cols = BLOCK_SIZE / 16
#define BLOCK_ROWS 16
#define BLOCK_COLS (BLOCK_SIZE / BLOCK_ROWS)

// Tiling parameter in the K-dimension
#define TILE_K 16

// This kernel treats the 3D tensor A (N x M x K) as a 2D matrix of shape (N*M x K).
// The matrix B is (K x L) and the output is (N*M x L), which is later reshaped as (N x M x L).
// Block dimensions are set to (BLOCK_COLS, BLOCK_ROWS). The kernel uses shared memory tiling
// and loads tiles cooperatively -- each thread loads multiple elements if required.

template <typename scalar_t>
__global__ void blocksize_experiment_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    const int num_rows,  // num_rows = N * M (flattened first two dims of A)
    const int K,
    const int L) {

    int row = blockIdx.y * BLOCK_ROWS + threadIdx.y;   // output row index
    int col = blockIdx.x * BLOCK_COLS + threadIdx.x;     // output column index

    scalar_t sum = 0;

    // Allocate shared memory tiles
    __shared__ scalar_t tile_A[BLOCK_ROWS][TILE_K];
    __shared__ scalar_t tile_B[TILE_K][BLOCK_COLS];

    // Number of tiles to cover the K dimension
    int numTiles = (K + TILE_K - 1) / TILE_K;
    
    for (int t = 0; t < numTiles; t++) {
        // Load tile from matrix A (of size BLOCK_ROWS x TILE_K) into shared memory.
        // Since blockDim.x may be smaller than TILE_K, loop over the tile columns in steps of BLOCK_COLS.
        for (int j = threadIdx.x; j < TILE_K; j += BLOCK_COLS) {
            int aCol = t * TILE_K + j;
            if (row < num_rows && aCol < K)
                tile_A[threadIdx.y][j] = A[row * K + aCol];
            else
                tile_A[threadIdx.y][j] = 0;
        }

        // Load tile from matrix B (of size TILE_K x BLOCK_COLS) into shared memory.
        // Use a loop over the tile rows because blockDim.y might be smaller than TILE_K.
        for (int i = threadIdx.y; i < TILE_K; i += BLOCK_ROWS) {
            int bRow = t * TILE_K + i;
            if (bRow < K && col < L)
                tile_B[i][threadIdx.x] = B[bRow * L + col];
            else
                tile_B[i][threadIdx.x] = 0;
        }

        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (int i = 0; i < TILE_K; i++) {
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the computed value to output if within bounds
    if (row < num_rows && col < L) {
        output[row * L + col] = sum;
    }
}

// CUDA forward function. It flattens the first two dimensions of A (N x M) to form a 2D matrix with shape (N*M x K).
void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);
    
    // Flatten N and M dimensions
    int num_rows = N * M;

    // Set up block and grid dimensions based on the experiment block size
    dim3 block(BLOCK_COLS, BLOCK_ROWS);
    dim3 grid((L + BLOCK_COLS - 1) / BLOCK_COLS, (num_rows + BLOCK_ROWS - 1) / BLOCK_ROWS);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "blocksize_experiment_kernel", ([&] {
        blocksize_experiment_kernel<scalar_t><<<grid, block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            num_rows, K, L);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in module_fn_cuda_forward: %s\n", cudaGetErrorString(err));
    }
}

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This function wraps the CUDA forward kernel
torch::Tensor module_fn_forward(
    torch::Tensor A,
    torch::Tensor B) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  
  int N = A.size(0);
  int M = A.size(1);
  int L = B.size(1);
  
  // Allocate output tensor with shape (N, M, L)
  auto output = torch::zeros({N, M, L}, A.options());
  module_fn_cuda_forward(A, B, output);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &module_fn_forward, "Block size experiment 3D tensor-matrix multiplication (CUDA)");
}
