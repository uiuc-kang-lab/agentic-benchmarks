#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 16

// This kernel performs batched matrix multiplication for a 3D tensor A (shape: N x M x K) multiplied by a 2D matrix B (shape: K x L),
// producing an output tensor of shape (N x M x L).  It uses tiling and shared memory to ensure that global memory accesses
// are coalesced. Each block computes a BLOCK_SIZE x BLOCK_SIZE tile of the output for a given batch (n).

template <typename scalar_t>
__global__ void tensor_matmul_tiled_kernel(
    const scalar_t* __restrict__ A,  // A: [N, M, K]
    const scalar_t* __restrict__ B,  // B: [K, L]
    scalar_t* __restrict__ C,        // C: [N, M, L]
    int M, int K, int L) {
  
  // Identify the output element to compute
  int row = blockIdx.y * BLOCK_SIZE + threadIdx.y; // index within M dimension
  int col = blockIdx.x * BLOCK_SIZE + threadIdx.x; // index within L dimension
  int n = blockIdx.z; // batch index

  scalar_t value = 0;

  // Loop over tiles of the K dimension
  int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
  
  // Declare shared memory for A and B tiles
  __shared__ scalar_t As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ scalar_t Bs[BLOCK_SIZE][BLOCK_SIZE];

  for (int t = 0; t < numTiles; t++) {
    // Compute column index for A tile
    int a_col = t * BLOCK_SIZE + threadIdx.x;
    // Load A tile element if within bounds; A has shape [N, M, K]
    if (row < M && a_col < K) {
      As[threadIdx.y][threadIdx.x] = A[n * M * K + row * K + a_col];
    } else {
      As[threadIdx.y][threadIdx.x] = 0;
    }

    // Compute row index for B tile
    int b_row = t * BLOCK_SIZE + threadIdx.y;
    // Load B tile element if within bounds; B has shape [K, L]
    if (b_row < K && col < L) {
      Bs[threadIdx.y][threadIdx.x] = B[b_row * L + col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();
    
    // Multiply the two tiles together
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE; i++) {
      value += As[threadIdx.y][i] * Bs[i][threadIdx.x];
    }
    __syncthreads();
  }

  // Write the computed value to the output tensor if within bounds
  if (row < M && col < L) {
    C[n * M * L + row * L + col] = value;
  }
}


void tensor_matmul_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {
  // A: [N, M, K], B: [K, L], output: [N, M, L]
  int N = A.size(0);
  int M = A.size(1);
  int K = A.size(2);
  int L = B.size(1);

  // Setup grid dimensions: each block computes a tile of size BLOCK_SIZE x BLOCK_SIZE for one batch index n
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((L + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE, N);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "tensor_matmul_tiled_cuda_forward", ([&] {
    tensor_matmul_tiled_kernel<scalar_t><<<grid, block>>>(
        A.data_ptr<scalar_t>(),
        B.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        M, K, L);
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in tensor_matmul_tiled_cuda_forward: %s\n", cudaGetErrorString(err));
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

  // Allocate the output tensor of shape [N, M, L]
  auto output = torch::zeros({N, M, L}, A.options());
  tensor_matmul_cuda_forward(A, B, output);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &tensor_matmul_forward, "Tiled tensor matrix multiplication (CUDA)");
}
