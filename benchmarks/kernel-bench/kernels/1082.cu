#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

// Define tile dimension
#define TILE_DIM 16

// New CUDA kernel using shared memory tiling for 3D tensor-matrix multiplication
// A has shape (N, M, K), B has shape (K, L) and output is (N, M, L).
// This kernel maps blockIdx.z to the batch dimension, blockIdx.y to the M dimension, and blockIdx.x to the L dimension.

template <typename scalar_t>
__global__ void efficient_3d_tiled_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int N, int M, int K, int L) {

    // Map the block and thread indices to output coordinates
    int n = blockIdx.z; // batch
    int m = blockIdx.y * TILE_DIM + threadIdx.y; // row in A's slice
    int l = blockIdx.x * TILE_DIM + threadIdx.x; // column in B

    scalar_t sum = 0;

    // Calculate how many tiles we need to cover the K dimension
    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;

    // Declare shared memory for tiles of A and B
    __shared__ scalar_t tileA[TILE_DIM][TILE_DIM];
    __shared__ scalar_t tileB[TILE_DIM][TILE_DIM];

    for (int t = 0; t < numTiles; ++t) {
        // Load a tile from A: each block processes one row portion from the A slice
        int tiledK_A = t * TILE_DIM + threadIdx.x;
        if (m < M && tiledK_A < K) {
            tileA[threadIdx.y][threadIdx.x] = A[n * M * K + m * K + tiledK_A];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0;
        }

        // Load a tile from B: each block processes one column portion from B
        int tiledK_B = t * TILE_DIM + threadIdx.y;
        if (tiledK_B < K && l < L) {
            tileB[threadIdx.y][threadIdx.x] = B[tiledK_B * L + l];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Compute the partial dot product for this tile
        #pragma unroll
        for (int k_inner = 0; k_inner < TILE_DIM; ++k_inner) {
            sum += tileA[threadIdx.y][k_inner] * tileB[k_inner][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the computed value to the output tensor if within bounds
    if (m < M && l < L) {
        output[n * M * L + m * L + l] = sum;
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

    // 3D grid: x covers L, y covers M, and z covers batch N
    dim3 block(TILE_DIM, TILE_DIM, 1);
    dim3 grid((L + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM, N);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "efficient_3d_tiled_kernel", ([&] {
        efficient_3d_tiled_kernel<scalar_t><<<grid, block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, M, K, L);
    }));

    // Check for errors during kernel launch
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

  auto output = torch::zeros({N, M, L}, A.options());
  module_fn_cuda_forward(A, B, output);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &module_fn_forward, "Efficient 3D Tensor-Matrix Multiplication with Tiling (CUDA)");
}
