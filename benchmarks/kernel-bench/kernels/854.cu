#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#define TILE_DIM 32
#define BLOCK_SIZE 16

// Helper function to load tiles of matrix A
__device__ __forceinline__
void loadTileA(const float* __restrict__ A, float tileA[TILE_DIM][TILE_DIM], int row, int tile, int M, int K) {
    int col = tile * TILE_DIM + threadIdx.x;
    if (row < M && col < K) {
        tileA[threadIdx.y][threadIdx.x] = A[row * K + col];
    } else {
        tileA[threadIdx.y][threadIdx.x] = 0.0f;
    }
}

// Helper function to load tiles of matrix B
__device__ __forceinline__
void loadTileB(const float* __restrict__ B, float tileB[TILE_DIM][TILE_DIM], int col, int tile, int K, int N) {
    int row = tile * TILE_DIM + threadIdx.y;
    if (row < K && col < N) {
        tileB[threadIdx.y][threadIdx.x] = B[row * N + col];
    } else {
        tileB[threadIdx.y][threadIdx.x] = 0.0f;
    }
}

// Device function for computing the partial result of a tile
__device__ __forceinline__
float computeTile(const float tileA[TILE_DIM][TILE_DIM], const float tileB[TILE_DIM][TILE_DIM]) {
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < TILE_DIM; i++) {
        sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
    }
    return sum;
}

// Optimized kernel using modular warp-aligned matrix multiplication
__global__ void optimized_matmul_kernel(const float* __restrict__ A,
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
        loadTileA(A, tileA, row, t, M, K);
        loadTileB(B, tileB, col, t, K, N);

        __syncthreads();
        
        sum += computeTile(tileA, tileB);
        
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host function to launch the optimized kernel
void matmul_optimized(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    if (M > 512 || N > 512 || K > 512) {
        cublasHandle_t handle;
        cublasCreate(&handle);

        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B.data_ptr<float>(), N, A.data_ptr<float>(), K, &beta, C.data_ptr<float>(), N);

        cublasDestroy(handle);
    } else {
        dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
        dim3 numBlocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

        optimized_matmul_kernel<<<numBlocks, threadsPerBlock>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }

        cudaDeviceSynchronize();
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", [](torch::Tensor A, torch::Tensor B) {
        TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
        TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
        auto C = torch::zeros({A.size(0), B.size(1)}, A.options());
        matmul_optimized(A, B, C);
        return C;
    }, "Optimized matrix multiplication (CUDA)");
}