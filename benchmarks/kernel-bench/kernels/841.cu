#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Define block and tile dimensions
#define BLOCK_SIZE 16
#define TILE_DIM (BLOCK_SIZE * 2)

// Kernel: Each thread computes a 2x2 block of C using register tiling.
__global__ void reg_tile_matmul_kernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int M, int K, int N) {
    int row0 = blockIdx.y * TILE_DIM + threadIdx.y;
    int col0 = blockIdx.x * TILE_DIM + threadIdx.x;
    int row1 = row0 + BLOCK_SIZE;
    int col1 = col0 + BLOCK_SIZE;

    float Cvalue00 = 0.0f, Cvalue01 = 0.0f, Cvalue10 = 0.0f, Cvalue11 = 0.0f;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    for (int tile = 0; tile < numTiles; tile++) {
        int tStart = tile * TILE_DIM;

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int aRow = blockIdx.y * TILE_DIM + threadIdx.y + i * BLOCK_SIZE;
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                int aCol = tStart + threadIdx.x + j * BLOCK_SIZE;
                int sharedRow = threadIdx.y + i * BLOCK_SIZE;
                int sharedCol = threadIdx.x + j * BLOCK_SIZE;
                if (aRow < M && aCol < K) {
                    As[sharedRow][sharedCol] = A[aRow * K + aCol];
                } else {
                    As[sharedRow][sharedCol] = 0.0f;
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int bRow = tStart + threadIdx.y + i * BLOCK_SIZE;
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                int bCol = blockIdx.x * TILE_DIM + threadIdx.x + j * BLOCK_SIZE;
                int sharedRow = threadIdx.y + i * BLOCK_SIZE;
                int sharedCol = threadIdx.x + j * BLOCK_SIZE;
                if (bRow < K && bCol < N) {
                    Bs[sharedRow][sharedCol] = B[bRow * N + bCol];
                } else {
                    Bs[sharedRow][sharedCol] = 0.0f;
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            float a_val0 = As[threadIdx.y][k];
            float a_val1 = As[threadIdx.y + BLOCK_SIZE][k];
            float b_val0 = Bs[k][threadIdx.x];
            float b_val1 = Bs[k][threadIdx.x + BLOCK_SIZE];
            Cvalue00 += a_val0 * b_val0;
            Cvalue01 += a_val0 * b_val1;
            Cvalue10 += a_val1 * b_val0;
            Cvalue11 += a_val1 * b_val1;
        }

        __syncthreads();
    }

    if (row0 < M && col0 < N) {
        C[row0 * N + col0] = Cvalue00;
    }
    if (row0 < M && col1 < N) {
        C[row0 * N + col1] = Cvalue01;
    }
    if (row1 < M && col0 < N) {
        C[row1 * N + col0] = Cvalue10;
    }
    if (row1 < M && col1 < N) {
        C[row1 * N + col1] = Cvalue11;
    }
}

// Host function to launch the kernel
void matmul_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Use cuBLAS for large matrices
    if (M > 512 || N > 512 || K > 512) {
        cublasHandle_t handle;
        cublasCreate(&handle);

        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B.data_ptr<float>(), N, A.data_ptr<float>(), K, &beta, C.data_ptr<float>(), N);

        cublasDestroy(handle);
    } else {
        // Grid dimensions: each block computes a TILE_DIM x TILE_DIM output tile
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

        reg_tile_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", [](torch::Tensor A, torch::Tensor B) {
        CHECK_INPUT(A);
        CHECK_INPUT(B);
        auto C = torch::zeros({A.size(0), B.size(1)}, A.options());
        matmul_cuda(A, B, C);
        return C;
    }, "Hybrid matrix multiplication (CUDA)");
}