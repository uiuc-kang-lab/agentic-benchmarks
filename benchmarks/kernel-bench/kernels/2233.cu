#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Tile dimensions for output (M and N) and partitioning of K
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

// This kernel partitions the K-dimension across a 3D grid. Each block computes a partial sum
// for a tile of the output matrix C. When multiple blocks (i.e., gridDim.z > 1) contribute to
// the same element of C, atomicAdd is used to safely accumulate the partial sums, thereby
// minimizing atomic operation usage only to the necessary final accumulation step.

__global__ void atomicTiledMatmulKernel(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int K, int M, int N) {
    // Compute the row and column for C computed by this thread
    int row = blockIdx.x * TILE_M + threadIdx.y;  // corresponds to row index in C (and A^T)
    int col = blockIdx.y * TILE_N + threadIdx.x;  // corresponds to col index in C (and B)
    
    // Determine the starting index for the current partition of K
    int k_start = blockIdx.z * TILE_K;

    float sum = 0.0f;

    // Only proceed if the thread maps to a valid output element
    if (row < M && col < N) {
        // Loop over the tile segment in the K-dimension
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            int global_k = k_start + k;
            if (global_k < K) {
                // A is of shape (K, M), so element A[k, row] is at A[k * M + row]
                // B is of shape (K, N), so element B[k, col] is at B[k * N + col]
                sum += __ldg(&A[global_k * M + row]) * __ldg(&B[global_k * N + col]);
            }
        }
        
        // If there's only one block along the K-dimension, write directly to C;
        // otherwise, use atomicAdd to accumulate the partial result.
        if (gridDim.z == 1) {
            C[row * N + col] = sum;
        } else {
            atomicAdd(&C[row * N + col], sum);
        }
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

    // Initialize output tensor C with zeros (M x N)
    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Define block and grid dimensions. We use a 3D grid to partition the K-dimension.
    dim3 blockDim(TILE_N, TILE_M);  // blockDim.x corresponds to output column, blockDim.y to output row
    dim3 gridDim((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N, (K + TILE_K - 1) / TILE_K);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    atomicTiledMatmulKernel<<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using atomic reduced tiling (CUDA)");
}
