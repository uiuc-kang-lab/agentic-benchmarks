#include <torch/extension.h>
#include <cuda_runtime.h>

// Define tile size for blocking
#define TILE_SIZE 16

// Device function to load a tile from matrix A into shared memory
__device__ __forceinline__ void loadTileA(const float* __restrict__ A, 
                                             float tileA[TILE_SIZE][TILE_SIZE], 
                                             int M, int K, 
                                             int blockRow, int tileIdx) {
    int row = blockRow + threadIdx.y;
    int col = tileIdx * TILE_SIZE + threadIdx.x;
    tileA[threadIdx.y][threadIdx.x] = (row < M && col < K) ? A[row * K + col] : 0.0f;
}

// Device function to load a tile from matrix B into shared memory
__device__ __forceinline__ void loadTileB(const float* __restrict__ B, 
                                             float tileB[TILE_SIZE][TILE_SIZE], 
                                             int K, int N, 
                                             int blockCol, int tileIdx) {
    int row = tileIdx * TILE_SIZE + threadIdx.y;
    int col = blockCol + threadIdx.x;
    tileB[threadIdx.y][threadIdx.x] = (row < K && col < N) ? B[row * N + col] : 0.0f;
}

// Device function to compute the product of loaded tiles
__device__ __forceinline__ float computeTileProduct(const float tileA[TILE_SIZE][TILE_SIZE], 
                                                        const float tileB[TILE_SIZE][TILE_SIZE]) {
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
        sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }
    return sum;
}

// Kernel function performing tiled matrix multiplication using modular device functions
__global__ void matmul_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int K, int N) {
    // Shared memory for storing sub-tiles of A and B
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float value = 0.0f;

    // Number of tiles to cover the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    int blockRow = blockIdx.y * TILE_SIZE;
    int blockCol = blockIdx.x * TILE_SIZE;

    // Loop over all tiles required to compute C[row, col]
    for (int t = 0; t < numTiles; t++) {
        loadTileA(A, tileA, M, K, blockRow, t);
        loadTileB(B, tileB, K, N, blockCol, t);
        __syncthreads();
        value += computeTileProduct(tileA, tileB);
        __syncthreads();
    }

    // Write the result if within bounds
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

// Helper macros to check tensor properties
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Forward function wrapper for the kernel launch
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE); grid.x = min(grid.x, (N + TILE_SIZE - 1) / TILE_SIZE); grid.y = min(grid.y, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular matrix multiplication using device functions (CUDA)");
}
