#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>

// Define tile size for our custom kernel
#define TILE_SIZE 16

// Macros to check inputs
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Branchless safe load functions for matrix A and B
__device__ inline float safeLoadA(const float *A, int row, int col, int M, int K) {
    int safe_row = (row < M) ? row : M - 1;
    int safe_col = (col < K) ? col : K - 1;
    float mask = (row < M && col < K) ? 1.0f : 0.0f;
    return A[safe_row * K + safe_col] * mask;
}

__device__ inline float safeLoadB(const float *B, int row, int col, int K, int N) {
    int safe_row = (row < K) ? row : K - 1;
    int safe_col = (col < N) ? col : N - 1;
    float mask = (row < K && col < N) ? 1.0f : 0.0f;
    return B[safe_row * N + safe_col] * mask;
}

// Custom CUDA kernel using tiling with branchless safe loads
__global__ void matmul_branchless_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    float sum = 0.0f;

    __shared__ float tileA[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tileB[TILE_SIZE + 1][TILE_SIZE];

    // Loop over tiles of matrix A and B
    for (int t = 0; t < K; t += TILE_SIZE) {
        // Load a tile of A and B with branchless safe loads
        tileA[ty][tx] = safeLoadA(A, row, t + tx, M, K);
        tileB[ty][tx] = safeLoadB(B, t + ty, col, K, N);
        __syncthreads();
        
        // Unrolled multiply-add for the tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = __fmaf_rn(tileA[ty][k], tileB[k][tx], sum);
        }
        __syncthreads();
    }

    // Write result if within bounds
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// Forward function that adapts the algorithm based on matrix sizes
// For small matrices, we use our custom kernel to avoid cuBLAS overhead,
// while for larger matrices, we leverage cuBLAS's highly optimized SGEMM implementation.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    // Validate inputs
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Allocate output tensor
    torch::Tensor C = torch::zeros({M, N}, A.options());

    // Adaptive selection threshold
    // For very small matrices, the kernel overhead can be minimized using our custom kernel.
    bool use_custom_kernel = (M < 64 || N < 64 || K < 64);

    if (use_custom_kernel) {
        // Launch our custom branchless tiling kernel
        dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
        dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        matmul_branchless_kernel<<<numBlocks, threadsPerBlock>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, N, K
        );
    } else {
        // Use cuBLAS SGEMM for larger matrices
        cublasHandle_t handle;
        cublasCreate(&handle);
        
        float alpha = 1.0f;
        float beta = 0.0f;
        // Note: cuBLAS expects column-major matrices. Since our tensors are row-major, we swap A and B
        // and also swap M and N in the call.
        cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,   // number of rows of matrix C in column-major layout
            M,   // number of columns of matrix C in column-major layout
            K,
            &alpha,
            B.data_ptr<float>(),   // B acts as the left operand
            N,
            A.data_ptr<float>(),   // A acts as the right operand
            K,
            &beta,
            C.data_ptr<float>(),
            N
        );
        
        cublasDestroy(handle);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive Matrix Multiplication (CUDA) using custom kernel for small inputs and cuBLAS for large ones");
}
