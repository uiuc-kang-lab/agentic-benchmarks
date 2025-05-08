#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// Device function to load a tile from matrix A into shared memory
__device__ inline void loadTileA(const float* A, float tileA[TILE_SIZE][TILE_SIZE], int M, int K, int blockRow, int tileIdx) {
    int row = blockRow + threadIdx.y;
    int col = tileIdx * TILE_SIZE + threadIdx.x;
    tileA[threadIdx.y][threadIdx.x] = (row < M && col < K) ? A[row * K + col] : 0.0f;
}

// Device function to load a tile from matrix B into shared memory
__device__ inline void loadTileB(const float* B, float tileB[TILE_SIZE][TILE_SIZE], int K, int N, int blockCol, int tileIdx) {
    int row = tileIdx * TILE_SIZE + threadIdx.y;
    int col = blockCol + threadIdx.x;
    tileB[threadIdx.y][threadIdx.x] = (row < K && col < N) ? B[row * N + col] : 0.0f;
}

// Kernel function performing tiled matrix multiplication using modular device functions
__global__ void matmul_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int K, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tileB[TILE_SIZE + 1][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float value = 0.0f;
    
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    int blockRow = blockIdx.y * TILE_SIZE;
    int blockCol = blockIdx.x * TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        loadTileA(A, tileA, M, K, blockRow, t);
        loadTileB(B, tileB, K, N, blockCol, t);
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

// The forward function wraps the kernel launch
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Tensor B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "Tensor A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "Tensor B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with minimized warp divergence (CUDA)");
}