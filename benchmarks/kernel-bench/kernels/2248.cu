#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Define tile and block dimensions
#define BLOCK_M 16        // Number of C-rows computed per block
#define BLOCK_N 32        // Number of C-columns computed per block (each thread computes 2 outputs)
#define TILE 16           // Tile width for the K dimension

// Kernel computes C = A.T * B, where A is (K, M), B is (K, N) and C is (M, N).
// Each thread computes two adjacent elements in C to improve reuse of loaded tiles.
__global__ void tiledDoubleOutputKernel(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int K, int M, int N) {
    // Map each block to a tile of C of size BLOCK_M x BLOCK_N.
    // Each thread computes two adjacent outputs in the horizontal (column) direction.
    int row = blockIdx.y * BLOCK_M + threadIdx.y;
    int col = blockIdx.x * BLOCK_N + threadIdx.x * 2;  // two outputs per thread

    float out0 = 0.0f, out1 = 0.0f;

    // Declare shared memory tiles for A and B
    // A_tile: holds a tile of A.T (which is logically A transposed). Each element is loaded as A[k, row] = A[k * M + row].
    __shared__ float A_tile[BLOCK_M][TILE];   // Dimensions: (BLOCK_M x TILE)
    // B_tile: holds a tile of B, dimensions: (TILE x BLOCK_N)
    __shared__ float B_tile[TILE][BLOCK_N];

    int numTiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; t++) {
        int tileStart = t * TILE;

        // Each thread loads one element of the A tile.
        int a_k = tileStart + threadIdx.x;  // threadIdx.x in [0, TILE-1]
        if (a_k < K && row < M)
            A_tile[threadIdx.y][threadIdx.x] = A[a_k * M + row];
        else
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;

        // Each thread loads two elements of the B tile.
        int b_k = tileStart + threadIdx.y;  // threadIdx.y in [0, TILE-1]
        int global_col0 = col;
        int global_col1 = col + 1;
        if (b_k < K) {
            if (global_col0 < N)
                B_tile[threadIdx.y][threadIdx.x * 2] = B[b_k * N + global_col0];
            else
                B_tile[threadIdx.y][threadIdx.x * 2] = 0.0f;
            if (global_col1 < N)
                B_tile[threadIdx.y][threadIdx.x * 2 + 1] = B[b_k * N + global_col1];
            else
                B_tile[threadIdx.y][threadIdx.x * 2 + 1] = 0.0f;
        } else {
            B_tile[threadIdx.y][threadIdx.x * 2] = 0.0f;
            B_tile[threadIdx.y][threadIdx.x * 2 + 1] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot-products for the two outputs
        #pragma unroll
        for (int s = 0; s < TILE; s++) {
            float a_val = A_tile[threadIdx.y][s];
            out0 += a_val * B_tile[s][threadIdx.x * 2];
            out1 += a_val * B_tile[s][threadIdx.x * 2 + 1];
        }

        __syncthreads();
    }

    // Write the computed outputs to global memory
    if (row < M) {
        if (col < N)
            C[row * N + col] = out0;
        if (col + 1 < N)
            C[row * N + col + 1] = out1;
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

    // Define block dimensions:
    // We use 16 threads for the row dimension and 16 threads for the column dimension,
    // with each thread computing two adjacent output elements (total BLOCK_N = 32 columns per block).
    dim3 blockDim(TILE, BLOCK_M); // blockDim.x = 16, blockDim.y = 16
    dim3 gridDim((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    tiledDoubleOutputKernel<<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using tiled kernel with double output per thread (CUDA)");
}
