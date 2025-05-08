#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Tile dimension for shared memory tiling
#define TILE_DIM 16

// CUDA kernel for computing C = A.T * B using tiling with shared memory.
// Here, A is of shape (K, M) and B is of shape (K, N). We interpret A.T as a matrix of shape (M, K),
// so that C[i, j] = sum_{k=0}^{K-1} A[k*M + i] * B[k*N + j].
// Each block computes a tile of C of size TILE_DIM x TILE_DIM.
__global__ void warpDivergenceOptimizedKernel(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               int K, int M, int N) {
    // Compute global row and column index for C.
    int row = blockIdx.x * TILE_DIM + threadIdx.y;  // row index in C (and in A^T)
    int col = blockIdx.y * TILE_DIM + threadIdx.x;  // column index in C

    float cValue = 0.0f;

    // Allocate shared memory for a tile of A_t and B
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    // Loop over tiles of the K dimension
    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < numTiles; t++) {
        // Load tile of A^T into shared memory.
        // A^T[i, k] is stored as A[k*M + i].
        int k_index = t * TILE_DIM + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && k_index < K) ? A[k_index * M + row] : 0.0f;

        // Load tile of B into shared memory.
        int k_index_b = t * TILE_DIM + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (col < N && k_index_b < K) ? B[k_index_b * N + col] : 0.0f;

        __syncthreads();

        // Perform the multiplication for this tile without divergent branches
        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            cValue += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result to C if within bounds
    if (row < M && col < N) {
        C[row * N + col] = cValue;
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

    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Set block size and grid dimensions
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 grid((M + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    warpDivergenceOptimizedKernel<<<grid, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using warp divergence optimization (CUDA)");
}
