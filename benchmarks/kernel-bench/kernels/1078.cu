#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

#define TILE_DIM 16

// CUDA kernel that minimizes warp divergence by separating full blocks from boundary blocks
// Each block computes a TILE_DIM x TILE_DIM tile of the flattened output matrix (N*M x L).

template <typename scalar_t>
__global__ void tiled_nodivergence_kernel(
    const scalar_t* __restrict__ A,  // A has shape (N, M, K), flattened as (N*M x K)
    const scalar_t* __restrict__ B,  // B has shape (K, L)
    scalar_t* __restrict__ output,   // output is (N*M x L), later reshaped to (N, M, L)
    const int num_rows,              // equals N * M
    const int K,
    const int L) {

    // Compute row and column indices within the output matrix
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    scalar_t sum = 0;

    // Determine if the current block is a full block (all threads load valid data)
    bool fullBlock = (blockIdx.x < gridDim.x - 1) && (blockIdx.y < gridDim.y - 1);

    __shared__ scalar_t tile_A[TILE_DIM][TILE_DIM];
    __shared__ scalar_t tile_B[TILE_DIM][TILE_DIM];

    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_DIM + threadIdx.x;
        int bRow = t * TILE_DIM + threadIdx.y;

        // For full blocks, no boundary check is needed.
        if (fullBlock) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + aCol];
            tile_B[threadIdx.y][threadIdx.x] = B[bRow * L + col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = (row < num_rows && aCol < K) ? A[row * K + aCol] : scalar_t(0);
            tile_B[threadIdx.y][threadIdx.x] = (bRow < K && col < L) ? B[bRow * L + col] : scalar_t(0);
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_DIM; i++) {
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result into global memory
    if (fullBlock) {
        output[row * L + col] = sum;
    } else {
        if (row < num_rows && col < L) {
            output[row * L + col] = sum;
        }
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

    int num_rows = N * M;  // Flatten first two dimensions of A

    // Define block and grid dimensions
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((L + TILE_DIM - 1) / TILE_DIM, (num_rows + TILE_DIM - 1) / TILE_DIM);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "tiled_nodivergence_kernel", ([&] {
        tiled_nodivergence_kernel<scalar_t><<<grid, block>>>(
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

// Macros for input checking
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// C++ interface wrapping the CUDA kernel launch
torch::Tensor module_fn_forward(
    torch::Tensor A,
    torch::Tensor B) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);

  int N = A.size(0);
  int M = A.size(1);
  int L = B.size(1);

  // Output tensor shaped as (N, M, L)
  auto output = torch::zeros({N, M, L}, A.options());
  module_fn_cuda_forward(A, B, output);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &module_fn_forward, "Tiled 3D Tensor-Matrix Multiplication with Reduced Warp Divergence (CUDA)");
}
