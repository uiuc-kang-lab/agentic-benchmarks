#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 16
#define TILE_DIM (BLOCK_SIZE * 2)
#define CUBLAS_THRESHOLD 512

__global__ void shared_memory_optimized_matmul_kernel(const float* __restrict__ A,
                                                      const float* __restrict__ B,
                                                      float* __restrict__ C,
                                                      int M, int K, int N) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float Cvalue = 0.0f;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        if (row < M && tile * BLOCK_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + tile * BLOCK_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && tile * BLOCK_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(tile * BLOCK_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Cvalue += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = Cvalue;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());

    if (M >= CUBLAS_THRESHOLD && N >= CUBLAS_THRESHOLD && K >= CUBLAS_THRESHOLD) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                   N, M, K, &alpha,
                   B.data_ptr<float>(), N,
                   A.data_ptr<float>(), K,
                   &beta, C.data_ptr<float>(), N);
        cublasDestroy(handle);
    } else {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
        shared_memory_optimized_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), 
                                                              B.data_ptr<float>(), 
                                                              C.data_ptr<float>(), 
                                                              M, K, N);
    }
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Shared memory optimized matrix multiplication (CUDA)");
}