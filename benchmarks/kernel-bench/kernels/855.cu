#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#define SMALL_BLOCK_SIZE 16
#define LARGE_BLOCK_SIZE 32
#define TILE_DIM_SMALL (SMALL_BLOCK_SIZE * 2)
#define TILE_DIM_LARGE (LARGE_BLOCK_SIZE * 2)
#define CUBLAS_THRESHOLD 512

__global__ void adaptive_matmul_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int K, int N, int block_size, int tile_dim) {
    int row0 = blockIdx.y * tile_dim + threadIdx.y;
    int col0 = blockIdx.x * tile_dim + threadIdx.x;
    int row1 = row0 + block_size;
    int col1 = col0 + block_size;

    float Cvalue00 = 0.0f, Cvalue01 = 0.0f, Cvalue10 = 0.0f, Cvalue11 = 0.0f;

    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = shared_mem + tile_dim * tile_dim;

    for (int tile = 0; tile < (K + tile_dim - 1) / tile_dim; tile++) {
        int tStart = tile * tile_dim;

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int aRow = blockIdx.y * tile_dim + threadIdx.y + i * block_size;
            for (int j = 0; j < 2; j++) {
                int aCol = tStart + threadIdx.x + j * block_size;
                int sharedRow = threadIdx.y + i * block_size;
                int sharedCol = threadIdx.x + j * block_size;
                As[sharedRow * tile_dim + sharedCol] = (aRow < M && aCol < K) ? 
                    A[aRow * K + aCol] : 0.0f;
            }
        }

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int bRow = tStart + threadIdx.y + i * block_size;
            for (int j = 0; j < 2; j++) {
                int bCol = blockIdx.x * tile_dim + threadIdx.x + j * block_size;
                int sharedRow = threadIdx.y + i * block_size;
                int sharedCol = threadIdx.x + j * block_size;
                Bs[sharedRow * tile_dim + sharedCol] = (bRow < K && bCol < N) ? 
                    B[bRow * N + bCol] : 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < tile_dim; k++) {
            float a_val0 = As[threadIdx.y * tile_dim + k];
            float a_val1 = As[(threadIdx.y + block_size) * tile_dim + k];
            float b_val0 = Bs[k * tile_dim + threadIdx.x];
            float b_val1 = Bs[k * tile_dim + threadIdx.x + block_size];
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

void matmul_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

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
        int block_size = (M < 256 && N < 256 && K < 256) ? SMALL_BLOCK_SIZE : LARGE_BLOCK_SIZE;
        int tile_dim = block_size * 2;
        dim3 block(block_size, block_size);
        dim3 grid((N + tile_dim - 1) / tile_dim, (M + tile_dim - 1) / tile_dim);
        size_t shared_mem_size = 2 * tile_dim * tile_dim * sizeof(float);
        adaptive_matmul_kernel<<<grid, block, shared_mem_size>>>(A.data_ptr<float>(), 
                                                                 B.data_ptr<float>(), 
                                                                 C.data_ptr<float>(), 
                                                                 M, K, N, block_size, tile_dim);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", [](torch::Tensor A, torch::Tensor B) {
        TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
        TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");
        auto C = torch::zeros({A.size(0), B.size(1)}, A.options());
        matmul_cuda(A, B, C);
        return C;
    }, "Adaptive block size matrix multiplication (CUDA)");
}