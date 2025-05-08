#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Define tile dimension optimized for 128-bit aligned accesses
#define TILE_DIM 16

// This kernel uses __ldg() for load instructions on read-only global memory
// The loads are assumed to be 128-bit aligned, improving bandwidth utilization.

template <typename scalar_t>
__global__ void aligned_ldg_tiled_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    const int N, const int M, const int K, const int L) {

    // Treat the output as 2D: (N*M) x L
    // Each row corresponds to one (M)-row from one batch of the 3D input tensor.
    int row = blockIdx.y * TILE_DIM + threadIdx.y; // overall row index
    int col = blockIdx.x * TILE_DIM + threadIdx.x; // overall column index

    // Derive batch index and intra-batch row index
    int batch = row / M;   // which batch slice
    int m = row % M;       // row index inside the current slice

    __shared__ scalar_t tile_A[TILE_DIM][TILE_DIM];
    __shared__ scalar_t tile_B[TILE_DIM][TILE_DIM];

    scalar_t sum = 0;
    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < numTiles; t++) {
        int a_col = t * TILE_DIM + threadIdx.x;
        int b_row = t * TILE_DIM + threadIdx.y;

        // Load one element of A from global memory with __ldg() for read-only access
        if (batch < N && m < M && a_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = __ldg(&A[batch * M * K + m * K + a_col]);
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0;
        }

        // Load one element of B using __ldg()
        if (b_row < K && col < L) {
            tile_B[threadIdx.y][threadIdx.x] = __ldg(&B[b_row * L + col]);
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Compute partial dot product for the tile
        #pragma unroll
        for (int i = 0; i < TILE_DIM; i++) {
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the computed value to global memory
    if (batch < N && m < M && col < L) {
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

    dim3 threads(TILE_DIM, TILE_DIM);
    // The grid is arranged so that each thread block computes a TILE_DIM x TILE_DIM tile of the (N*M)xL output
    dim3 grid((L + TILE_DIM - 1) / TILE_DIM, ((N * M) + TILE_DIM - 1) / TILE_DIM);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "aligned_ldg_tiled_kernel", ([&] {
        aligned_ldg_tiled_kernel<scalar_t><<<grid, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, M, K, L);
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
  m.def("forward", &module_fn_forward, "Aligned __ldg Read-Optimized 3D Tensor-Matrix Multiplication (CUDA)");
}
