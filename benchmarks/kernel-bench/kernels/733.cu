#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>

// Macros to check input
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Tile size for the custom kernel
#define TILE_WIDTH 16

// Custom tiled matrix multiplication kernel with shared memory and branchless predicated loads
__global__ void MatmulKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N) {
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

// Hybrid forward function: selects the optimal path based on problem size
// For small matrices, use the custom kernel; for larger ones, leverage highly optimized cuBLAS routines.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    // Validate inputs
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // Heuristic threshold: use custom kernel for small matrices
    // Tweak these thresholds for your specific use-case and hardware
    bool useCustomKernel = (M <= 256 && N <= 256 && K <= 256);

    if (useCustomKernel) {
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
        dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
        
        // Launch custom tiled kernel
        MatmulKernel<<<gridDim, blockDim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    } else {
        // Use cuBLAS for larger matrices
        cublasHandle_t handle;
        cublasStatus_t status = cublasCreate(&handle);
        TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "Failed to create cuBLAS handle");

        float alpha = 1.0f;
        float beta = 0.0f;

        // cuBLAS assumes column-major order. Since our tensors are in row-major order (PyTorch),
        // we swap the order of the matrices and perform: C^T = B^T * A^T.
        status = cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             B.data_ptr<float>(), N,
                             A.data_ptr<float>(), K,
                             &beta,
                             C.data_ptr<float>(), N);
        TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cuBLAS sgemm failed");

        cublasDestroy(handle);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid matrix multiplication (CUDA) with adaptive kernel selection");
}
