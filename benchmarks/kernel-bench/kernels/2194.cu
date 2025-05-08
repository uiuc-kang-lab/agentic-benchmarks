#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Tile dimensions and block unroll factor
#define TILE_M 16
#define TILE_N 16
#define BLOCK_K 32

// This kernel computes C = A.T * B for A of shape (K, M) and B of shape (K, N) with full float32 precision.
// It uses shared memory tiling and loop unrolling while minimizing warp divergence through uniform control flow.

__global__ void tiledSharedUnrollNoDivKernel(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               int K, int M, int N) {
    // Each block computes a TILE_M x TILE_N tile of C.
    int row = blockIdx.x * TILE_M + threadIdx.x;  // Corresponds to index i of C
    int col = blockIdx.y * TILE_N + threadIdx.y;  // Corresponds to index j of C

    float sum = 0.0f;

    // Loop over K in chunks of BLOCK_K
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // Compute effective tile size for K in this iteration
        int tileK = (K - k0) >= BLOCK_K ? BLOCK_K : (K - k0);

        // Declare shared memory tiles.
        __shared__ float As[BLOCK_K][TILE_M];
        __shared__ float Bs[BLOCK_K][TILE_N];

        // Use all threads in the block to load the shared memory tiles in a uniform manner
        int tid = threadIdx.y * TILE_M + threadIdx.x;
        int totalThreads = TILE_M * TILE_N;

        // Load tile from A into shared memory.
        // A is stored as (K, M): element A[k, i] is at A[k * M + i]
        for (int index = tid; index < tileK * TILE_M; index += totalThreads) {
            int t = index / TILE_M;  // index in the K-tile
            int i = index % TILE_M;  // index within the tile in M dimension
            int global_i = blockIdx.x * TILE_M + i;
            int global_k = k0 + t;
            // Avoid divergent branching using a ternary operator; this is compiled as predicated instructions
            float a_val = (global_i < M && global_k < K) ? A[global_k * M + global_i] : 0.0f;
            As[t][i] = a_val;
        }

        // Load tile from B into shared memory.
        // B is stored as (K, N): element B[k, j] is at B[k * N + j]
        for (int index = tid; index < tileK * TILE_N; index += totalThreads) {
            int t = index / TILE_N;  // index in the K-tile
            int j = index % TILE_N;  // index within the tile in N dimension
            int global_j = blockIdx.y * TILE_N + j;
            int global_k = k0 + t;
            float b_val = (global_j < N && global_k < K) ? B[global_k * N + global_j] : 0.0f;
            Bs[t][j] = b_val;
        }

        __syncthreads();

        // Compute partial dot product for this tile.
        // To avoid divergent branches, we separate the full-tile and partial-tile cases uniformly.
        if (tileK == BLOCK_K) {
            // Unroll loop by factor of 4 in the full tile case.
            #pragma unroll
            for (int t = 0; t < BLOCK_K; t += 4) {
                sum += As[t][threadIdx.x]   * Bs[t][threadIdx.y]
                     + As[t+1][threadIdx.x] * Bs[t+1][threadIdx.y]
                     + As[t+2][threadIdx.x] * Bs[t+2][threadIdx.y]
                     + As[t+3][threadIdx.x] * Bs[t+3][threadIdx.y];
            }
        } else {
            #pragma unroll
            for (int t = 0; t < tileK; t++) {
                sum += As[t][threadIdx.x] * Bs[t][threadIdx.y];
            }
        }

        __syncthreads(); // Ensure all threads have completed this tile before proceeding
    }

    // Write the result back to global memory if within bounds; this conditional is executed uniformly per thread.
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// The forward function exposed via PyBind11.
// A: Tensor of shape (K, M) [CUDA, float32]
// B: Tensor of shape (K, N) [CUDA, float32]
// Returns: Tensor C of shape (M, N) computed as C = A.T * B

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

    // Define thread block and grid dimensions based on tile sizes
    dim3 block(TILE_M, TILE_N);
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    tiledSharedUnrollNoDivKernel<<<grid, block>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using tiled shared memory with unrolling and minimal warp divergence (CUDA)");
}
