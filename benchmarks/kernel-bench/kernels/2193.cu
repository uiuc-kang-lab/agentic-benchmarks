#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Tile dimensions and unroll factor
#define TILE_M 16
#define TILE_N 16
#define BLOCK_K 32

// Kernel to compute C = A.T * B using tiled shared memory with loop unrolling
// A: shape (K, M), B: shape (K, N), C: shape (M, N) computed as C[i, j] = sum_{k=0}^{K-1} A[k*M + i] * B[k*N + j]
__global__ void tiledSharedUnrollKernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int K, int M, int N) {
    // Global row and column indices for C
    int row = blockIdx.x * TILE_M + threadIdx.x;  // corresponds to index i
    int col = blockIdx.y * TILE_N + threadIdx.y;  // corresponds to index j

    float sum = 0.0f;

    // Allocate shared memory for a tile of A and B
    // As: stores a tile of A for current k-chunk. Dimensions: BLOCK_K x TILE_M
    // Bs: stores a tile of B for current k-chunk. Dimensions: BLOCK_K x TILE_N
    __shared__ float As[BLOCK_K][TILE_M];
    __shared__ float Bs[BLOCK_K][TILE_N];

    // Loop over k dimension in increments of BLOCK_K
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // Each block needs to load BLOCK_K * TILE_M elements for A and BLOCK_K * TILE_N elements for B
        int tid = threadIdx.y * blockDim.x + threadIdx.x;  // Unique thread index within the block
        int totalThreads = blockDim.x * blockDim.y;         // should be TILE_M * TILE_N

        // Load tile of A into shared memory
        // A is stored in row-major order with shape (K, M): element A[k, i] is at A[k * M + i]
        for (int index = tid; index < BLOCK_K * TILE_M; index += totalThreads) {
            int t = index / TILE_M;       // local k index within the tile
            int i = index % TILE_M;         // local i index within the tile
            int global_i = blockIdx.x * TILE_M + i;
            int global_k = k0 + t;
            if (global_i < M && global_k < K)
                As[t][i] = A[global_k * M + global_i];
            else
                As[t][i] = 0.0f;
        }

        // Load tile of B into shared memory
        // B is stored in row-major order with shape (K, N): element B[k, j] is at B[k * N + j]
        for (int index = tid; index < BLOCK_K * TILE_N; index += totalThreads) {
            int t = index / TILE_N;       // local k index within the tile
            int j = index % TILE_N;         // local j index within the tile
            int global_j = blockIdx.y * TILE_N + j;
            int global_k = k0 + t;
            if (global_j < N && global_k < K)
                Bs[t][j] = B[global_k * N + global_j];
            else
                Bs[t][j] = 0.0f;
        }

        __syncthreads();  // Ensure the shared memory tiles are loaded before computation

        // Compute the partial dot product for this k-chunk using loop unrolling
        #pragma unroll
        for (int t = 0; t < BLOCK_K; t += 4) {
            sum += As[t][threadIdx.x] * Bs[t][threadIdx.y];
            if (t + 1 < BLOCK_K) sum += As[t + 1][threadIdx.x] * Bs[t + 1][threadIdx.y];
            if (t + 2 < BLOCK_K) sum += As[t + 2][threadIdx.x] * Bs[t + 2][threadIdx.y];
            if (t + 3 < BLOCK_K) sum += As[t + 3][threadIdx.x] * Bs[t + 3][threadIdx.y];
        }

        __syncthreads();  // Synchronize before loading the next tile
    }

    // Write the result to global memory if within bounds
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// PyBind11 forward function exposed to Python
// A: Tensor of shape (K, M) [CUDA, float32]
// B: Tensor of shape (K, N) [CUDA, float32]
// Returns: C, Tensor of shape (M, N) computed as C = A.T * B

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

    // Define block and grid dimensions
    dim3 block(TILE_M, TILE_N);  // 16x16 threads per block
    dim3 grid((M + TILE_M - 1) / TILE_M,
              (N + TILE_N - 1) / TILE_N);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Launch the kernel
    tiledSharedUnrollKernel<<<grid, block>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using tiled shared memory with unrolling (CUDA)");
}
