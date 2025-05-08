#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_SIZE 16

// Device function to load a tile of matrix A into shared memory
__device__ inline void loadTileA(const float* __restrict__ A, float tileA[TILE_SIZE][TILE_SIZE], int M, int K, int row, int tile_col_offset) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = tile_col_offset + tx;
    if(row < M && col < K)
        tileA[ty][tx] = A[row * K + col];
    else
        tileA[ty][tx] = 0.0f;
}

// Device function to load a tile of matrix B into shared memory
__device__ inline void loadTileB(const float* B, float tileB[TILE_SIZE][TILE_SIZE], int N, int K, int tile_row_offset, int col) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = tile_row_offset + ty;
    if(row < K && col < N)
        tileB[ty][tx] = B[row * N + col];
    else
        tileB[ty][tx] = 0.0f;
}

// Device function to compute the dot product for the current tile
__device__ inline float computeTileSum(const float tileA[TILE_SIZE][TILE_SIZE], const float tileB[TILE_SIZE][TILE_SIZE]) {
    float partial_sum = 0.0f;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
        partial_sum += tileA[ty][k] * tileB[k][tx];
    }
    return partial_sum;
}

__global__ void matmul_modular_kernel(const float* __restrict__ A, const float* B, float* C, int M, int N, int K) {
    // Compute the row and column index of the C element
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    
    // Allocate shared memory for tiles of A and B
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Loop over tiles of the K dimension
    for (int tile_offset = 0; tile_offset < K; tile_offset += TILE_SIZE) {
        // Load one tile of A and B into shared memory using modular device functions
        loadTileA(A, tileA, M, K, row, tile_offset);
        loadTileB(B, tileB, N, K, tile_offset, col);
        __syncthreads();

        // Compute partial sum for the tile
        sum += computeTileSum(tileA, tileB);
        __syncthreads();
    }

    // Write the result back to global memory
    if (row < M && col < N)
        C[row * N + col] = sum;
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}, A.options());

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_modular_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular Matrix Multiplication (CUDA)");
}
