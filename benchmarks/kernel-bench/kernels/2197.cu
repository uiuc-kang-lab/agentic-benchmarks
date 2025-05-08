#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Tile dimensions and chunk size
#define TILE_M 16  // Tile width corresponding to output rows (and A's column index)
#define TILE_N 16  // Tile height corresponding to output columns (and B's column index)
#define BLOCK_K 32 // Chunk size along the K dimension

// This kernel computes C = A.T * B, where A is of shape (K, M) and B is of shape (K, N).
// The output C is of shape (M, N) with C[i, j] = sum_{k=0}^{K-1} A[k * M + i] * B[k * N + j].
// To ensure memory coalescing, the kernel loads tiles of A and B into shared memory with properly
// aligned access, so that consecutive threads in a warp access consecutive global memory locations.

__global__ void tiledSharedCoalescedKernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int K, int M, int N) {
    // Compute the global indices for the output matrix C
    int i = blockIdx.x * TILE_M + threadIdx.x; // row in C (and column in A since C = A.T * B)
    int j = blockIdx.y * TILE_N + threadIdx.y; // column in C (and column in B)

    float sum = 0.0f;

    // Shared memory tiles for A and B
    // For A: We load a tile of size BLOCK_K x TILE_M
    // For B: We load a tile of size BLOCK_K x TILE_N
    __shared__ float shared_A[BLOCK_K][TILE_M];
    __shared__ float shared_B[BLOCK_K][TILE_N];

    // Loop over K in chunks of size BLOCK_K
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // Load a tile of A from global memory into shared memory.
        // A is stored as row-major with shape (K, M): element A[k, i] is at A[k * M + i].
        // We want coalesced access: for fixed k, consecutive i are contiguous in memory.
        // Each thread loads multiple elements if needed.
        for (int index = threadIdx.x + threadIdx.y * TILE_M; index < BLOCK_K * TILE_M; index += TILE_M * TILE_N) {
            int local_k = index / TILE_M;
            int local_i = index % TILE_M;
            int global_i = blockIdx.x * TILE_M + local_i;
            int global_k = k0 + local_k;
            if (global_k < K && global_i < M)
                shared_A[local_k][local_i] = A[global_k * M + global_i];
            else
                shared_A[local_k][local_i] = 0.0f;
        }

        // Load a tile of B from global memory into shared memory.
        // B is stored as row-major with shape (K, N): element B[k, j] is at B[k * N + j].
        // For coalescing, we let consecutive threads load consecutive j indices.
        // We rearrange the thread index such that local_j varies fastest.
        for (int index = threadIdx.x + threadIdx.y * TILE_N; index < BLOCK_K * TILE_N; index += TILE_M * TILE_N) {
            int local_k = index / TILE_N;
            int local_j = index % TILE_N;
            int global_j = blockIdx.y * TILE_N + local_j;
            int global_k = k0 + local_k;
            if (global_k < K && global_j < N)
                shared_B[local_k][local_j] = B[global_k * N + global_j];
            else
                shared_B[local_k][local_j] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product for this block's current k-chunk
        // Each thread computes its output element using the shared memory tiles.
        #pragma unroll
        for (int t = 0; t < BLOCK_K; t++) {
            // For C[i, j] computation, we need A[k, i] from shared_A and B[k, j] from shared_B.
            sum += shared_A[t][threadIdx.x] * shared_B[t][threadIdx.y];
        }

        __syncthreads();
    }

    // Write the computed value to C if within bounds
    if (i < M && j < N) {
        C[i * N + j] = sum;
    }
}

// The forward function exposed via PyBind11. It ensures that the inputs are CUDA float32 tensors
// and then launches the tiledSharedCoalescedKernel to compute C = A.T * B with coalesced global
// memory accesses for improved performance.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    // Dimensions: A: (K, M) and B: (K, N). The result C = A.T * B will have shape (M, N).
    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Define block and grid dimensions
    // Each block computes a TILE_M x TILE_N tile of C
    dim3 block(TILE_M, TILE_N);
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Launch the CUDA kernel
    tiledSharedCoalescedKernel<<<grid, block>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using tiled shared memory with coalesced global accesses (CUDA)");
}
