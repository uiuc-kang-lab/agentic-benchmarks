#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// The BLOCK_DIM macro determines the square block size (threads per block = BLOCK_DIM * BLOCK_DIM).
// To experiment with different total block sizes (e.g., 32, 64, 128, 256, 512), adjust BLOCK_DIM accordingly.
#ifndef BLOCK_DIM
#define BLOCK_DIM 16  // default: 16x16 = 256 threads/block
#endif

// Kernel for 3D tensor (A: [N, M, K]) and matrix (B: [K, L]) multiplication.
// Output is [N, M, L]. The kernel uses shared memory tiling with a tile width equal to BLOCK_DIM.
// Each thread computes one element of the output by accumulating partial dot products over K in tiles.

template <typename scalar_t>
__global__ void experiment_block_size_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    const int N, const int M, const int K, const int L) {

    // BLOCK_DIM is the tile dimension (both rows and columns).
    const int block_dim = BLOCK_DIM;

    // Compute a flattened output row index for A, where A is treated as a 2D tensor of shape (N*M, K).
    int global_row = blockIdx.y * block_dim + threadIdx.y;  // corresponds to (n * M + m)
    int col = blockIdx.x * block_dim + threadIdx.x;         // corresponds to L dimension

    // Derive batch index and intra-batch row index (m) from global_row
    int batch = global_row / M;  
    int m = global_row % M;

    scalar_t sum = 0;

    // Allocate shared memory for tiles of A and B
    __shared__ scalar_t tile_A[BLOCK_DIM][BLOCK_DIM];  // tile from A
    __shared__ scalar_t tile_B[BLOCK_DIM][BLOCK_DIM];  // tile from B

    // Loop over tiles along the K dimension
    for (int t = 0; t < K; t += block_dim) {
        int tiled_col = t + threadIdx.x;  // Column index within A's row
        int tiled_row = t + threadIdx.y;   // Row index for B

        // Load element from A into shared memory if within bounds
        if (global_row < N * M && tiled_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[batch * M * K + m * K + tiled_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0;
        }

        // Load element from B into shared memory if within bounds
        if (tiled_row < K && col < L) {
            tile_B[threadIdx.y][threadIdx.x] = B[tiled_row * L + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (int i = 0; i < block_dim; i++) {
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result to global memory if within bounds
    if (global_row < N * M && col < L) {
        output[batch * M * L + m * L + col] = sum;
    }
}

// CUDA forward function
void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    const int N = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int L = B.size(1);

    int block_dim = BLOCK_DIM;
    // The output is a 2D tensor of shape (N*M, L).
    dim3 threads(block_dim, block_dim);
    dim3 grid((L + block_dim - 1) / block_dim, ((N * M) + block_dim - 1) / block_dim);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "experiment_block_size_kernel", ([&] {
        experiment_block_size_kernel<scalar_t><<<grid, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, M, K, L);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in experiment_block_size_kernel: %s\n", cudaGetErrorString(err));
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

  auto N = A.size(0);
  auto M = A.size(1);
  auto L = B.size(1);

  auto output = torch::zeros({N, M, L}, A.options());
  module_fn_cuda_forward(A, B, output);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &module_fn_forward, "Experiment with block sizes for 3D tensor-matrix multiplication (CUDA)");
}
