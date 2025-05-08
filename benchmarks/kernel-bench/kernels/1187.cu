#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 32
#define WARP_SIZE 32

template <typename scalar_t>
__global__ void tensor_matmul_optimized_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int M, int K, int L) {
  
  // Indices for block-tile calculations
  int m_idx = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  int l_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int n_idx = blockIdx.z;

  __shared__ scalar_t As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ scalar_t Bs[BLOCK_SIZE][BLOCK_SIZE];

  scalar_t sum = 0;
  int num_tiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for (int t = 0; t < num_tiles; ++t) {
    int k_idx = t * BLOCK_SIZE;

    // Load A tile with vectorization-friendly access
    int a_row = m_idx;
    int a_col = k_idx + threadIdx.x;
    if (a_row < M && a_col < K)
      As[threadIdx.y][threadIdx.x] = A[n_idx * M * K + a_row * K + a_col];
    else
      As[threadIdx.y][threadIdx.x] = 0;

    // Load B tile with optimized transpose access
    int b_row = k_idx + threadIdx.y;
    int b_col = l_idx;
    if (b_row < K && b_col < L)
      Bs[threadIdx.x][threadIdx.y] = B[b_row * L + b_col];
    else
      Bs[threadIdx.x][threadIdx.y] = 0;

    __syncthreads();

    // Warp-wide reduction with shuffle instructions
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      scalar_t a_val = As[threadIdx.y][i];
      scalar_t b_val = Bs[threadIdx.x][i];
      sum += a_val * b_val;
    }
    
    __syncthreads();
  }

  // Final warp-level reduction
  for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }

  if (m_idx < M && l_idx < L) {
    if (threadIdx.x == 0) {
      C[n_idx * M * L + m_idx * L + l_idx] = sum;
    }
  }
}

void tensor_matmul_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {
  int N = A.size(0);
  int M = A.size(1);
  int K = A.size(2);
  int L = B.size(1);

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((L + BLOCK_SIZE - 1)/BLOCK_SIZE, 
            (M + BLOCK_SIZE - 1)/BLOCK_SIZE,
            N);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "tensor_matmul_cuda_forward", ([&] {
    tensor_matmul_optimized_kernel<scalar_t><<<grid, block>>>(
        A.data_ptr<scalar_t>(),
        B.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        M, K, L);
  }));
}

TORCH_LIBRARY(tiled_matmul, m) {
  m.def("forward", &tensor_matmul_cuda_forward);
}