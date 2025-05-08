#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel
template <typename scalar_t>
__global__ void module_fn_cuda_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int N, int M, int K, int L) {

    int n = blockIdx.z;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.x * blockDim.x + threadIdx.x;

    if (m < M && l < L) {
    scalar_t sum = 0;
    const int TILE_SIZE = 32; // assuming blockDim.x == 32
    // Loop over tiles of the k-dimension
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Allocate shared memory for a tile of B
        __shared__ scalar_t B_tile[TILE_SIZE][32 + 1]; // pad second dimension to avoid bank conflicts

        // Cooperative loading of B tile: each thread loads multiple elements
        int index = threadIdx.y * blockDim.x + threadIdx.x;
        int num_threads = blockDim.x * blockDim.y;
        for (int i = index; i < TILE_SIZE * blockDim.x; i += num_threads) {
            int tile_row = i / blockDim.x;
            int col_in_tile = i % blockDim.x;
            int k_val = tile * TILE_SIZE + tile_row;
            int global_col = blockIdx.x * blockDim.x + col_in_tile;
            if (k_val < K && global_col < L) {
                B_tile[tile_row][col_in_tile] = B[k_val * L + global_col];
            } else {
                B_tile[tile_row][col_in_tile] = 0;
            }
        }
        __syncthreads();

        // Compute partial dot product for this tile
        for (int i = 0; i < TILE_SIZE; ++i) {
            int k_val = tile * TILE_SIZE + i;
            if (k_val < K) {
                scalar_t a_val = A[n * M * K + m * K + k_val];
                scalar_t b_val = B_tile[i][threadIdx.x];
                sum += a_val * b_val;
            }
        }
        __syncthreads();
    }
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

    const int threads_x = 32;
    const int threads_y = 32;
    const dim3 threads(threads_x, threads_y);
    const dim3 blocks((L + threads_x - 1) / threads_x, (M + threads_y - 1) / threads_y, N);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "module_fn_cuda_forward", ([&] {
      module_fn_cuda_kernel<scalar_t><<<blocks, threads>>>(        
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
#define CHECK_CONTIGUOUS(x)  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)  CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

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
  m.def("forward", &module_fn_forward, "module_fn forward (CUDA)");
}