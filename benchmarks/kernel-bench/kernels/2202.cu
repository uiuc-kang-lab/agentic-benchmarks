#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Tile dimensions and chunk size
#define TILE_M 16
#define TILE_N 16
#define BLOCK_K 32

// This CUDA kernel computes C = A.T * B where:
//   A: shape (K, M) [read-only]
//   B: shape (K, N) [read-only]
//   C: shape (M, N) with C[i,j] = sum_{k=0}^{K-1} A[k*M + i] * B[k*N + j]
// It uses shared memory tiling and loop unrolling, and optimizes global memory loads
// by employing __ldg() to use the read-only cache. Global loads are assumed to be
// 128-bit aligned to maximize throughput.
__global__ void optimizedTiledMatMulKernel(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            int K, int M, int N) {
    // Compute global output indices
    int row = blockIdx.x * TILE_M + threadIdx.x;   // Corresponds to output row (and A column index)
    int col = blockIdx.y * TILE_N + threadIdx.y;   // Corresponds to output column of C

    float sum = 0.0f;

    // Allocate shared memory for current tile from A and B
    __shared__ float As[BLOCK_K][TILE_M];
    __shared__ float Bs[BLOCK_K][TILE_N];

    // Unique thread index in the block to coordinate global loads
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * blockDim.y;  // Should equal TILE_M * TILE_N

    // Loop over the K dimension in chunks
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // Load a tile of A into shared memory
        for (int index = tid; index < BLOCK_K * TILE_M; index += totalThreads) {
            int t = index / TILE_M;    // local index in K tile
            int m = index % TILE_M;      // local index in M dimension
            int global_k = k0 + t;
            int global_m = blockIdx.x * TILE_M + m;
            As[t][m] = (global_k < K && global_m < M) ? __ldg(&A[global_k * M + global_m]) : 0.0f;
        }

        // Load a tile of B into shared memory
        for (int index = tid; index < BLOCK_K * TILE_N; index += totalThreads) {
            int t = index / TILE_N;    // local index in K tile
            int n = index % TILE_N;      // local index in N dimension
            int global_k = k0 + t;
            int global_n = blockIdx.y * TILE_N + n;
            Bs[t][n] = (global_k < K && global_n < N) ? __ldg(&B[global_k * N + global_n]) : 0.0f;
        }

        __syncthreads(); // Ensure entire tile is loaded

        // Compute partial dot product for this tile with loop unrolling
        #pragma unroll
        for (int t = 0; t < BLOCK_K; t++) {
            sum += As[t][threadIdx.x] * Bs[t][threadIdx.y];
        }

        __syncthreads(); // Prepare for next tile load
    }

    // Write the computed value to output if within bounds
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Forward function exposed to PyTorch via PyBind11
// A: Tensor with shape (K, M) [CUDA, float32]
// B: Tensor with shape (K, N) [CUDA, float32]
// Returns: C with shape (M, N) computed as C = A.T * B
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

    // Define block and grid dimensions based on tile sizes
    dim3 block(TILE_M, TILE_N);
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    optimizedTiledMatMulKernel<<<grid, block>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using optimized tiled shared memory (CUDA)");
}
