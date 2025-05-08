#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_DIM 32

// Device function to load a tile from matrix A into shared memory
__device__ __forceinline__
void loadTileA(const float* __restrict__ A, float tileA[TILE_DIM][TILE_DIM], int row, int tile, int M, int K) {
    int col = tile * TILE_DIM + threadIdx.x;
    if (row < M && col < K) {
        tileA[threadIdx.y][threadIdx.x] = A[row * K + col];
    } else {
        tileA[threadIdx.y][threadIdx.x] = 0.0f;
    }
}

// Device function to load a tile from matrix B into shared memory
__device__ __forceinline__
void loadTileB(const float* __restrict__ B, float tileB[TILE_DIM][TILE_DIM], int col, int tile, int K, int N) {
    int row = tile * TILE_DIM + threadIdx.y;
    if (row < K && col < N) {
        tileB[threadIdx.y][threadIdx.x] = B[row * N + col];
    } else {
        tileB[threadIdx.y][threadIdx.x] = 0.0f;
    }
}

// Device function to compute the partial dot product from loaded tiles
__device__ __forceinline__
float computeTile(const float tileA[TILE_DIM][TILE_DIM], const float tileB[TILE_DIM][TILE_DIM]) {
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < TILE_DIM; i++) {
        sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
    }
    return sum;
}

// Modular warp-aligned matrix multiplication kernel
__global__ void modular_warp_aligned_matmul_kernel(const float* __restrict__ A,
                                                     const float* __restrict__ B,
                                                     float* __restrict__ C,
                                                     int M, int K, int N) {
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    
    float sum = 0.0f;

    __shared__ float tileA[TILE_DIM][TILE_DIM];
    __shared__ float tileB[TILE_DIM][TILE_DIM];

    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < numTiles; t++) {
        // Load a tile of A and B into shared memory using helper functions
        loadTileA(A, tileA, row, t, M, K);
        loadTileB(B, tileB, col, t, K, N);

        __syncthreads();
        
        // Compute partial result for this tile
        sum += computeTile(tileA, tileB);
        
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host function to launch the kernel
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}, A.options());

    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 numBlocks((N + TILE_DIM - 1) / TILE_DIM,
                   (M + TILE_DIM - 1) / TILE_DIM);

    modular_warp_aligned_matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Modular warp-aligned matrix multiplication (CUDA)");
}
