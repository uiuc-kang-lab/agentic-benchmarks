#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TILE_WIDTH 16

// Hybrid tiled matrix multiplication kernel using cuBLAS for larger matrices
__global__ void TiledMatmulKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N) {
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float cValue = 0.0f;

    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    
    for (int t = 0; t < numTiles; t++) {
        int tiledCol = t * TILE_WIDTH + threadIdx.x;
        int tiledRow = t * TILE_WIDTH + threadIdx.y;

        float aElem = (row < M && tiledCol < K) ? A[row * K + tiledCol] : 0.0f;
        float bElem = (tiledRow < K && col < N) ? B[tiledRow * N + col] : 0.0f;

        As[threadIdx.y][threadIdx.x] = aElem;
        Bs[threadIdx.y][threadIdx.x] = bElem;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i++) {
            cValue += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = cValue;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // Use cuBLAS for large matrices
    if (M > 512 || N > 512 || K > 512) {
        cublasHandle_t handle;
        cublasCreate(&handle);

        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B.data_ptr<float>(), N, A.data_ptr<float>(), K, &beta, C.data_ptr<float>(), N);

        cublasDestroy(handle);
    } else {
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
        dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

        TiledMatmulKernel<<<gridDim, blockDim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    }
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid optimized matrix multiplication (CUDA) using cuBLAS and tiled kernel");
}