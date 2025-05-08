#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// Utility functions implemented as device functions
__device__ __forceinline__ void load_tile_A(const float* __restrict__ A,
                                              float sA[TILE_SIZE][TILE_SIZE],
                                              int row, int m, int N) {
    int col_index = m * TILE_SIZE + threadIdx.x;
    sA[threadIdx.y][threadIdx.x] = (col_index < N && row >= col_index) ? A[row * N + col_index] : 0.0f;
}

__device__ __forceinline__ void load_tile_B(const float* __restrict__ B,
                                              float sB[TILE_SIZE][TILE_SIZE],
                                              int col, int m, int N) {
    int row_index = m * TILE_SIZE + threadIdx.y;
    sB[threadIdx.y][threadIdx.x] = (row_index < N && row_index >= col) ? B[row_index * N + col] : 0.0f;
}

__device__ __forceinline__ float compute_tile_sum(const float sA[TILE_SIZE][TILE_SIZE],
                                                    const float sB[TILE_SIZE][TILE_SIZE],
                                                    int row, int col, int tile_start, int tile_end) {
    float tile_sum = 0.0f;
    #pragma unroll
    for (int k = max(col, tile_start); k < min(row + 1, tile_end); ++k) {
        int local_k = k - tile_start;
        tile_sum += sA[threadIdx.y][local_k] * sB[local_k][threadIdx.x];
    }
    return tile_sum;
}

// Main kernel that performs lower triangular matrix multiplication with improved warp divergence handling
__global__ void warp_divergence_optimized_triangular_mm_kernel(const float* __restrict__ A,
                                                                const float* __restrict__ B,
                                                                float* __restrict__ C,
                                                                int N) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    if (row >= N || col >= N) return;

    float sum = 0.0f;
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    // Avoid divergent branching by processing unnecessary calculations
    const int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int m = 0; m < numTiles; m++) {
        load_tile_A(A, sA, row, m, N);
        load_tile_B(B, sB, col, m, N);
        __syncthreads();

        int tile_start = m * TILE_SIZE;
        int tile_end = tile_start + TILE_SIZE;
        if (tile_end > N) tile_end = N;

        sum += compute_tile_sum(sA, sB, row, col, tile_start, tile_end);
        __syncthreads();
    }

    C[row * N + col] = (row >= col) ? sum : 0.0f; // Unified control flow
}

// PyTorch interface function
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "A and B must be square matrices");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    warp_divergence_optimized_triangular_mm_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Warp Divergence Optimized Triangular Matrix Multiplication (CUDA)");
}
