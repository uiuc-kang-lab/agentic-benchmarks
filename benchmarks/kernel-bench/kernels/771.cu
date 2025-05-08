/*
Hybrid GEMM implementation combining a custom tiled kernel with branchless safe loads and cuBLAS for optimized performance
depending on problem size.
*/

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>

#define TILE_SIZE 16

// Macros for input validation
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Custom branchless safe load functions to avoid divergent if conditions
__device__ inline float safeLoadA(const float *A, int row, int col, int M, int K) {
    int safe_row = (row < M) ? row : (M - 1);
    int safe_col = (col < K) ? col : (K - 1);
    float mask = (row < M && col < K) ? 1.0f : 0.0f;
    return A[safe_row * K + safe_col] * mask;
}

__device__ inline float safeLoadB(const float *B, int row, int col, int K, int N) {
    int safe_row = (row < K) ? row : (K - 1);
    int safe_col = (col < N) ? col : (N - 1);
    float mask = (row < K && col < N) ? 1.0f : 0.0f;
    return B[safe_row * N + safe_col] * mask;
}

// Custom tiled kernel using shared memory and branchless indexing
__global__ void matmul_branchless_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Loop over tiles along the K dimension
    for (int t = 0; t < K; t += TILE_SIZE) {
        tileA[ty][tx] = safeLoadA(A, row, t + tx, M, K);
        tileB[ty][tx] = safeLoadB(B, t + ty, col, K, N);
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = __fmaf_rn(tileA[ty][k], tileB[k][tx], sum);
        }
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// Launch function for the custom kernel
void launch_custom_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_branchless_kernel<<<numBlocks, threadsPerBlock>>>(A, B, C, M, N, K);
}

// Launch function for cuBLAS based matrix multiplication
void launch_cublas_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Note: PyTorch tensors are in row-major order while cuBLAS expects column-major order.
    // Swapping A and B in the cublasSgemm call accounts for this layout difference.
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B, N,
                A, K,
                &beta,
                C, N);
    
    cublasDestroy(handle);
}

// Hybrid forward function that selects the optimal implementation based on matrix sizes
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}, A.options());

    // Heuristic: The custom kernel has lower overhead for small matrices, whereas cuBLAS is optimized for large ones.
    // If any dimension is smaller than 128, use the custom kernel; otherwise, switch to cuBLAS.
    if (M < 128 || N < 128 || K < 128) {
        launch_custom_matmul(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    } else {
        launch_cublas_matmul(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid Matrix Multiplication (CUDA)");
}
