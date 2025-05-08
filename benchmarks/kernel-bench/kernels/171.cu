/*
 * Hybrid GEMM: Uses a custom kernel for small matrices and cublasSgemm for large matrices
 * to reduce kernel launch overhead on small problems and leverage library optimizations on large ones.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <iostream>

// Tile size for the custom kernel
#define TILE_SIZE 32
// Threshold: if the total number of operations is smaller than this, use the custom kernel
#define CUBLAS_THRESHOLD 100000

// Input checking macro
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)              \
    CHECK_CUDA(x);                \
    CHECK_CONTIGUOUS(x);          \
    TORCH_CHECK(x.scalar_type() == torch::kFloat, #x " must be a float tensor")

// Custom matrix multiplication kernel using shared memory tiling and manual unrolling
__global__ void matrixMultiplyKernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       const int M, const int N, const int K) {
    // Add padding to avoid bank conflicts (32 banks on modern GPUs)
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];  // +1 padding to avoid bank conflicts
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load A tile
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + t * TILE_SIZE + threadIdx.x]);
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        // Load B tile
        if (t * TILE_SIZE + threadIdx.y < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = __ldg(&B[(t * TILE_SIZE + threadIdx.y) * N + col]);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial product for this tile, unrolling the loop in steps of 8
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k += 8) {
            sum += As[threadIdx.y][k]     * Bs[k][threadIdx.x];
            sum += As[threadIdx.y][k + 1] * Bs[k + 1][threadIdx.x];
            sum += As[threadIdx.y][k + 2] * Bs[k + 2][threadIdx.x];
            sum += As[threadIdx.y][k + 3] * Bs[k + 3][threadIdx.x];
            sum += As[threadIdx.y][k + 4] * Bs[k + 4][threadIdx.x];
            sum += As[threadIdx.y][k + 5] * Bs[k + 5][threadIdx.x];
            sum += As[threadIdx.y][k + 6] * Bs[k + 6][threadIdx.x];
            sum += As[threadIdx.y][k + 7] * Bs[k + 7][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Hybrid matrix multiplication: chooses custom kernel or cublasSgemm based on matrix dimensions
void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    // Decide based on the total number of operations
    if ((M * K * N) < CUBLAS_THRESHOLD) {
        // Launch the custom kernel for small matrix sizes
        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                    (M + TILE_SIZE - 1) / TILE_SIZE);
        matrixMultiplyKernel<<<blocks, threads>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, N, K
        );
    } else {
        // For larger matrices, use cuBLAS for optimized performance
        float *d_A = A.data_ptr<float>();
        float *d_B = B.data_ptr<float>();
        float *d_C = C.data_ptr<float>();

        cublasHandle_t handle;
        cublasCreate(&handle);
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Note: The input matrices are in row-major order. We swap arguments to emulate GEMM
        cublasStatus_t status = cublasSgemm(handle,
                                            CUBLAS_OP_N, CUBLAS_OP_N,
                                            N, M, K,
                                            &alpha,
                                            d_B, N,
                                            d_A, K,
                                            &beta,
                                            d_C, N);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "CUBLAS sgemm failed" << std::endl;
        }
        cublasDestroy(handle);
    }
}

// PyTorch binding: creates the output tensor and calls the hybrid multiply routine
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());
    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid Matrix multiplication (CUDA)");
}
