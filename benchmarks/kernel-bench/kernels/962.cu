#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define TILE_K 16
#define MATRIX_SIZE_THRESHOLD 1024

__device__ __forceinline__ float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    return transpose ? matrix[col * ld + row] : matrix[row * ld + col];
}

__global__ void matmul_kernel_standard(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int N, int K,
                                     int lda, int ldb, int ldc,
                                     bool transA, bool transB) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float C_value = 0.0f;
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        int tiled_k = t * BLOCK_SIZE;
        bool valid_A = (row < M && tiled_k + threadIdx.x < K);
        bool valid_B = (col < N && tiled_k + threadIdx.y < K);

        As[threadIdx.y][threadIdx.x] = valid_A ? get_element(A, row, tiled_k + threadIdx.x, lda, transA) : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = valid_B ? get_element(B, tiled_k + threadIdx.y, col, ldb, transB) : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            C_value += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * ldc + col] = C_value;
}

__global__ void matmul_kernel_atomic(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K,
                                   int lda, int ldb, int ldc,
                                   bool transA, bool transB) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int k_tile = blockIdx.z * TILE_K;

    __shared__ float As[BLOCK_SIZE][TILE_K];
    __shared__ float Bs[TILE_K][BLOCK_SIZE];
    float sum = 0.0f;

    if (row < M && (k_tile + threadIdx.x) < K) {
        As[threadIdx.y][threadIdx.x] = get_element(A, row, k_tile + threadIdx.x, lda, transA);
    }
    if (col < N && (k_tile + threadIdx.y) < K) {
        Bs[threadIdx.y][threadIdx.x] = get_element(B, k_tile + threadIdx.y, col, ldb, transB);
    }

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE_K; ++k) {
        sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    if (row < M && col < N) {
        atomicAdd(&C[row * ldc + col], sum);
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("Input tensors must be on CUDA devices");
    }

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);
    bool transA = false, transB = false;
    int lda = A.stride(0), ldb = B.stride(0), ldc = N;

    auto C = torch::empty({M, N}, A.options());
    
    bool use_atomic = (M > MATRIX_SIZE_THRESHOLD && N <= MATRIX_SIZE_THRESHOLD) || 
                     (N > MATRIX_SIZE_THRESHOLD && M <= MATRIX_SIZE_THRESHOLD);

    if (use_atomic) {
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (K + TILE_K - 1) / TILE_K);
        C.zero_();
        matmul_kernel_atomic<<<gridDim, blockDim>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
            M, N, K, lda, ldb, ldc, transA, transB);
    } else {
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
        matmul_kernel_standard<<<gridDim, blockDim>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
            M, N, K, lda, ldb, ldc, transA, transB);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Hybrid matrix multiplication (CUDA)");
}