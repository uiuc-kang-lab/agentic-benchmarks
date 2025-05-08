#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Define tile size for shared memory tiling
#define TILE_SIZE 16

// CUDA kernel for computing C = A.T * B using shared memory tiling.
// A: shape (K, M), B: shape (K, N), C: shape (M, N).
// Note: A.T(i,k) = A(k,i), so we load A in a transposed manner from global memory.
__global__ void matMulSharedKernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int K, int M, int N) {
    // Compute the row (i) and column (j) index in the output matrix C
    int row = blockIdx.x * TILE_SIZE + threadIdx.y;  // corresponds to i in C (and A's column index)
    int col = blockIdx.y * TILE_SIZE + threadIdx.x;  // corresponds to j in C (and B's column index)

    float sum = 0.0f;

    // Allocate shared memory for tiles of A and B
    // For A, we load elements from A in a transposed manner: A_t(row, k) = A(k, row).
    __shared__ float tileA[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tileB[TILE_SIZE + 1][TILE_SIZE];

    // Loop over the tiles of the k-dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        // Compute global index for A and load from global memory into shared memory tile.
        // For A_t: element (row, t*TILE_SIZE + threadIdx.x) equals A(t*TILE_SIZE + threadIdx.x, row)
        int aIndex = t * TILE_SIZE + threadIdx.x;
        if (row < M && aIndex < K) {
            tileA[threadIdx.y][threadIdx.x] = A[aIndex * M + row];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile for B directly from global memory.
        // B is stored as (K, N): element (t*TILE_SIZE + threadIdx.y, col) 
        int bIndex = t * TILE_SIZE + threadIdx.y;
        if (bIndex < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[bIndex * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Perform the multiplication for the tile
        #pragma unroll
        for (int k_inner = 0; k_inner < TILE_SIZE; k_inner++) {
            sum += tileA[threadIdx.y][k_inner] * tileB[k_inner][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the computed value to C if within valid indices
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// The forward function exposed via PyBind11.
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

    // Dimensions: A is (K, M) and B is (K, N).
    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    // Allocate output tensor C of shape (M, N).
    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Define thread block and grid sizes using tiling.
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Get raw pointers to tensor data.
    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Launch the shared memory tiled CUDA kernel.
    matMulSharedKernel<<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B (CUDA) using shared memory tiling");
}
