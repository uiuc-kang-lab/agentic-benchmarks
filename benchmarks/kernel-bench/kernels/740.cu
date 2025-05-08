#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Macro for experimenting with block sizes: Try 32, 64, 128, 256, or 512
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// Threshold for using the custom kernel vs cuBLAS fallback
#define MATRIX_SIZE_THRESHOLD 512

// This kernel assigns one thread per output element. For each element, the kernel loops
// over the small K dimension with unrolling to improve performance.
__global__ void TunedElementwiseMatmulKernel(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               int M, int K, int N) {
    // Calculate row and column indices for C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Forward function that dispatches the custom kernel for small matrices and falls back to cuBLAS otherwise.
// The block configuration can be tuned via the BLOCK_SIZE macro to experiment with different thread block sizes.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // For matrices with small M and N, use the custom kernel with configurable block size
    if (M <= MATRIX_SIZE_THRESHOLD && N <= MATRIX_SIZE_THRESHOLD) {
        // Experiment with different block sizes by setting BLOCK_SIZE (e.g., 32, 64, 128, 256, 512).
        // Here we choose blockDim.x = 16 if possible, and compute blockDim.y accordingly.
        int blockSize = BLOCK_SIZE;
        int blockDimX = (blockSize >= 16) ? 16 : 8;
        int blockDimY = blockSize / blockDimX;  // Ensure blockDim.x * blockDim.y == BLOCK_SIZE
        
        dim3 blockDim(blockDimX, blockDimY);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                     (M + blockDim.y - 1) / blockDim.y);
                      
        TunedElementwiseMatmulKernel<<<gridDim, blockDim>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, K, N);
    } else {
        // For larger matrices, fall back on the highly optimized cuBLAS
        static cublasHandle_t handle = nullptr;
        if (handle == nullptr) {
            cublasCreate(&handle);
        }
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    B.data_ptr<float>(), N,
                    A.data_ptr<float>(), K,
                    &beta, C.data_ptr<float>(), N);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tuned block size matrix multiplication (CUDA)");
}
