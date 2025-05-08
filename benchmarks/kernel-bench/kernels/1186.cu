#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 16

// This kernel performs batched matrix multiplication for a 3D tensor A (shape: N x M x K) and a 2D matrix B (shape: K x L),
// producing an output tensor of shape (N x M x L). It uses a tiled approach with shared memory. To minimize warp divergence,
// we avoid divergent branches in the tile-loading phase by computing safe indices and using arithmetic masks.

template <typename scalar_t>
__global__ void tensor_matmul_tiled_nodiv_kernel(
    const scalar_t* __restrict__ A,  // A: [N, M, K]
    const scalar_t* __restrict__ B,  // B: [K, L]
    scalar_t* __restrict__ C,        // C: [N, M, L]
    int M, int K, int L) {

  // Identify batch index and the row, col position for the output tile
  int n = blockIdx.z;
  int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  scalar_t value = 0;
  int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Shared memory tiles for A and B
  __shared__ scalar_t As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ scalar_t Bs[BLOCK_SIZE][BLOCK_SIZE];

  for (int t = 0; t < numTiles; t++) {
    // Compute global column index for A tile
    int a_col = t * BLOCK_SIZE + threadIdx.x;
    // Compute safe indices for A: if out-of-bound, clamp to the last valid index
    int safe_row = (row < M) ? row : (M - 1);
    int safe_a_col = (a_col < K) ? a_col : (K - 1);
    // Use an arithmetic mask: 1 if indices are within bounds, 0 otherwise
    int maskA = (row < M && a_col < K) ? 1 : 0;
    // Load tile element from A for batch n; A is laid out as [N, M, K]
    As[threadIdx.y][threadIdx.x] = maskA * A[n * M * K + safe_row * K + safe_a_col];

    // For B, calculate row index within the tile
    int b_row = t * BLOCK_SIZE + threadIdx.y;
    int safe_b_row = (b_row < K) ? b_row : (K - 1);
    int maskB = (b_row < K && col < L) ? 1 : 0;
    // Load tile element from B; B is [K, L]
    Bs[threadIdx.y][threadIdx.x] = maskB * B[safe_b_row * L + col];

    __syncthreads();

    // Multiply the two tiles together
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE; i++) {
      value = fma(As[threadIdx.y][i], Bs[i][threadIdx.x], value);
    }
    __syncthreads();
  }

  // Avoid out-of-bound writes; only threads with valid (row, col) write their computed value
  if (row < M && col < L) {
    C[n * M * L + row * L + col] = value;
  }
}


void tensor_matmul_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C) {
  // A: [N, M, K], B: [K, L], C: [N, M, L]
  int N = A.size(0);
  int M = A.size(1);
  int K = A.size(2);
  int L = B.size(1);

  // Setup grid and block dimensions
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((L + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE, N);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "tensor_matmul_tiled_nodiv_cuda_forward", ([&] {
    tensor_matmul_tiled_nodiv_kernel<scalar_t><<<grid, block>>>(
        A.data_ptr<scalar_t>(),
        B.data_ptr<scalar_t>(),
        C.data_ptr<scalar_t>(),
        M, K, L);
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in tensor_matmul_tiled_nodiv_cuda_forward: %s\n", cudaGetErrorString(err));
  }
}


// Helper macros to check tensors
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor tensor_matmul_forward(
    torch::Tensor A,
    torch::Tensor B) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);

  auto N = A.size(0);
  auto M = A.size(1);
  auto L = B.size(1);
  
  // Allocate output tensor of shape [N, M, L]
  auto C = torch::zeros({N, M, L}, A.options());
  tensor_matmul_cuda_forward(A, B, C);
  return C;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &tensor_matmul_forward, "Tiled tensor matrix multiplication with minimized warp divergence (CUDA)");
}
