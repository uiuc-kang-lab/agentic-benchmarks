#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to compute one dot product element for transposed matrices
template <typename scalar_t>
__device__ inline scalar_t compute_transposed_dot(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    const int row,
    const int col,
    const int M,
    const int K) {
  scalar_t sum = 0;
  for (int k = 0; k < K; ++k) {
    // A is transposed: stored as (K x M): element A[k * M + row]
    // B is transposed: stored as (N x K): element B[col * K + k]
    sum += A[k * M + row] * B[col * K + k];
  }
  return sum;
}

// CUDA kernel for matrix multiplication with transposed matrices using modular device function
template <typename scalar_t>
__global__ void matmul_transpose_kernel_modular(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,  // Number of rows in output
    const int N,  // Number of columns in output
    const int K) { // Shared dimension length

  // Identify row and column for the thread in the output matrix
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N) {
    // Use the modular device function to compute the element
    C[row * N + col] = compute_transposed_dot<scalar_t>(A, B, row, col, M, K);
  }
}

// Host function to launch the modular CUDA kernel
torch::Tensor matmul_transpose_cuda_modular(torch::Tensor A, torch::Tensor B) {
  // Dimensions:
  // A is transposed: shape (K, M)
  // B is transposed: shape (N, K)
  const int K = A.size(0);
  const int M = A.size(1);
  const int N = B.size(0);

  // Create output tensor C with shape (M, N)
  auto C = torch::empty({M, N}, A.options());

  // Define block and grid dimensions
  const int BLOCK_SIZE = 16;
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

  AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_kernel_modular", ([&] {
    matmul_transpose_kernel_modular<scalar_t><<<blocks, threads>>>(
        A.data_ptr<scalar_t>(),
        B.data_ptr<scalar_t>(),
        C.data_ptr<scalar_t>(),
        M, N, K);
  }));

  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &matmul_transpose_cuda_modular, "Modular Matrix multiplication with transposed matrices forward (CUDA)");
}
