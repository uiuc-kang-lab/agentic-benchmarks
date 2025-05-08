#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 32
#define TILE_DIM (BLOCK_SIZE * 2)
#define CUBLAS_THRESHOLD 512

__global__ void block_size_optimized_matmul_kernel(const float* __restrict__ A,
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

    for (int tile = 0; tile < (K + TILE_DIM - 1) / TILE_DIM; tile++) {
        int tStart = tile * TILE_DIM;

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int aRow = blockIdx.y * TILE_DIM + threadIdx.y + i * BLOCK_SIZE;
            for (int j = 0; j < 2; j++) {
                int aCol = tStart + threadIdx.x + j * BLOCK_SIZE;
                int sharedRow = threadIdx.y + i * BLOCK_SIZE;
                int sharedCol = threadIdx.x + j * BLOCK_SIZE;
                As[sharedRow][sharedCol] = (aRow < M && aCol < K) ? 
                    A[aRow * K + aCol] : 0.0f;
            }
        }

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int bRow = tStart + threadIdx.y + i * BLOCK_SIZE;
            for (int j = 0; j < 2; j++) {
                int bCol = blockIdx.x * TILE_DIM + threadIdx.x + j * BLOCK_SIZE;
                int sharedRow = threadIdx.y + i * BLOCK_SIZE;
                int sharedCol = threadIdx.x + j * BLOCK_SIZE;
                Bs[sharedRow][sharedCol] = (bRow < K && bCol < N) ? 
                    B[bRow * N + bCol] : 0.0f;
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

    if (row0 < M && col0 < N) C[row0 * N + col0] = Cvalue00;
    if (row0 < M && col1 < N) C[row0 * N + col1] = Cvalue01;
    if (row1 < M && col0 < N) C[row1 * N + col0] = Cvalue10;
    if (row1 < M && col1 < N) C[row1 * N + col1] = Cvalue11;
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
        dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
        block_size_optimized_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), 
                                                           B.data_ptr<float>(), 
                                                           C.data_ptr<float>(), 
                                                           M, K, N);
    }
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Block size optimized matrix multiplication (CUDA)");
}