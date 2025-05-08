/*
 * Combined Efficient Triangular Matrix Multiplication Kernel
 *
 * This kernel combines the best aspects from two approaches. It uses a helper function
 * to load tiles into shared memory (improving code clarity and coalesced memory access)
 * and leverages manual loop unrolling with effective boundary handling for triangular
 * matrices. It also sets the cache configuration to prefer L1 for better performance.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define WARP_SIZE 32

// Helper functions for loading tiles with triangular conditions
__device__ __forceinline__ void load_A_tile(float shTile[TILE_SIZE][TILE_SIZE],
                                              const float* __restrict__ A,
                                              const int row,
                                              const int a_col,
                                              const int N) {
    // For lower triangular matrix A, only load if a_col <= row
    if (a_col < N && a_col <= row)
        shTile[threadIdx.y][threadIdx.x] = A[row * N + a_col];
    else
        shTile[threadIdx.y][threadIdx.x] = 0.0f;
}

__device__ __forceinline__ void load_B_tile(float shTile[TILE_SIZE][TILE_SIZE],
                                              const float* __restrict__ B,
                                              const int b_row,
                                              const int col,
                                              const int N) {
    // For lower triangular matrix B, load only if b_row >= col
    if (b_row < N && b_row >= col)
        shTile[threadIdx.y][threadIdx.x] = B[b_row * N + col];
    else
        shTile[threadIdx.y][threadIdx.x] = 0.0f;
}

// Combined kernel: performs efficient triangular matrix multiplication
__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       const int N) {
    __shared__ float shA[TILE_SIZE][TILE_SIZE];
    __shared__ float shB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Boundary check and early exit for upper-triangular region
    if (row >= N || col >= N) return;
    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;

    // Compute the range of tiles that contribute to C[row, col]
    const int t_start = col / TILE_SIZE;
    const int t_end   = row / TILE_SIZE;

    // Iterate over the tiles
    #pragma unroll
    for (int t = t_start; t <= t_end; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        load_A_tile(shA, A, row, a_col, N);

        int b_row = t * TILE_SIZE + threadIdx.y;
        load_B_tile(shB, B, b_row, col, N);

        __syncthreads();

        // Compute the effective k range within this tile
        int k_start = max(t * TILE_SIZE, col);
        int k_end   = min((t + 1) * TILE_SIZE, row + 1);
        int effective_tile = k_end - k_start;

        // Use loop unrolling when the entire tile is used
        if (effective_tile == TILE_SIZE) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; k++) {
                sum += shA[threadIdx.y][k] * shB[k][threadIdx.x];
            }
        } else {
            // For boundary tiles, use the adjusted range
            for (int k = k_start; k < k_end; k++) {
                int local_k = k - t * TILE_SIZE;
                sum += shA[threadIdx.y][local_k] * shB[local_k][threadIdx.x];
            }
        }

        __syncthreads();
    }

    // Write the result to global memory
    C[row * N + col] = sum;
}

// C++ interface exposed to PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "A and B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Prefer L1 cache for improved performance
    cudaFuncSetCacheConfig(triangular_mm_kernel, cudaFuncCachePreferL1);

    triangular_mm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Combined efficient triangular matrix multiplication (CUDA)");
}
