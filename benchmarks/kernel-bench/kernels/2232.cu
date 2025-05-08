#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Define tile dimension and split factor for the K dimension
#define TILE_DIM 16
#define SPLIT_K 2  // Number of splits along the K dimension

// CUDA kernel for computing C = A.T * B using split-K and minimal atomic operations.
// Each block processes a slice of the K dimension. Partial results are accumulated into the final
// result using atomicAdd. This minimizes atomic operations to only one per output element per block.

__global__ void splitKMatmulKernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int K, int M, int N) {
    // Determine the range of K for this block based on the split factor
    int block_k_size = (K + SPLIT_K - 1) / SPLIT_K;  // size of each K-split
    int k_start = blockIdx.z * block_k_size;
    int k_end = (k_start + block_k_size < K) ? (k_start + block_k_size) : K;

    // Compute global row and column indices for C
    int row = blockIdx.x * TILE_DIM + threadIdx.y;  // Corresponds to row in C (and A^T)
    int col = blockIdx.y * TILE_DIM + threadIdx.x;    // Corresponds to col in C

    float cValue = 0.0f;

    // Shared memory for tiles of A and B
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    // Number of tiles required to cover the local slice in K
    int local_k = k_end - k_start;
    int numTiles = (local_k + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < numTiles; t++) {
        // Index in K dimension for A loading
        int k_idx = t * TILE_DIM + threadIdx.x;
        int global_k = k_start + k_idx;
        if (row < M && global_k < k_end) {
            // A is stored as (K, M) so element A[k, row] is at A[global_k * M + row]
            As[threadIdx.y][threadIdx.x] = A[global_k * M + row];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Index in K dimension for B loading
        int k_idx_b = t * TILE_DIM + threadIdx.y;
        global_k = k_start + k_idx_b;
        if (col < N && global_k < k_end) {
            // B is stored as (K, N) so element B[global_k, col] is at B[global_k * N + col]
            Bs[threadIdx.y][threadIdx.x] = B[global_k * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            cValue += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Accumulate the partial results into C using atomicAdd to handle race conditions from split-K
    if (row < M && col < N) {
        atomicAdd(&C[row * N + col], cValue);
    }
}

// The forward function exposed via PyBind11
// Inputs:
//   A: Tensor of shape (K, M) [CUDA, float32]
//   B: Tensor of shape (K, N) [CUDA, float32]
// Returns:
//   C: Tensor of shape (M, N) computed as A.T * B.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    // Allocate output tensor C and initialize to zero since we accumulate partial results using atomicAdd
    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((M + TILE_DIM - 1) / TILE_DIM,
                 (N + TILE_DIM - 1) / TILE_DIM,
                 SPLIT_K);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    splitKMatmulKernel<<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using split-K with minimal atomic operations (CUDA)");
}
