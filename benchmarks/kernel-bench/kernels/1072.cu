#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

#define TILE_DIM 16

// CUDA kernel using tiling and loop unrolling to improve performance
template <typename scalar_t>
__global__ void unrolled_tile_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int num_rows,  // equals N * M
    int K,
    int L) {

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    scalar_t sum = 0;

    __shared__ scalar_t tile_A[TILE_DIM][TILE_DIM];
    __shared__ scalar_t tile_B[TILE_DIM][TILE_DIM];

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
        int aCol = t * TILE_DIM + threadIdx.x;
        if (row < num_rows && aCol < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0;
        }

        int bRow = t * TILE_DIM + threadIdx.y;
        if (bRow < K && col < L) {
            tile_B[threadIdx.y][threadIdx.x] = B[bRow * L + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_DIM; i++) {
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }

        __syncthreads();
    }

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
    
    int num_rows = N * M;

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((L + TILE_DIM - 1) / TILE_DIM, (num_rows + TILE_DIM - 1) / TILE_DIM);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "unrolled_tile_cuda_forward", ([&] {
      unrolled_tile_kernel<scalar_t><<<grid, block>>>(
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

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This function wraps the CUDA forward function
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
  m.def("forward", &module_fn_forward, "unrolled tiling with memory coalescing (CUDA)");
}
