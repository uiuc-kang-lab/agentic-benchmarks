#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define WARPSIZE 32

__device__ float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    if (transpose)
        return matrix[col * ld + row];
    else
        return matrix[row * ld + col];
}

__global__ void matmul_kernel_warp_primitive(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              int M, int N, int K,
                                              int lda, int ldb, int ldc,
                                              bool transA, bool transB) {
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    int row = block_row * BLOCK_SIZE + thread_row;
    int col = block_col * BLOCK_SIZE + thread_col;

    float C_value = 0.0f;

    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        int tiled_k = t * BLOCK_SIZE;
        float A_value = (row < M && tiled_k + thread_col < K) ? get_element(A, row, tiled_k + thread_col, lda, transA) : 0.0f;
        float B_value = (col < N && tiled_k + thread_row < K) ? get_element(B, tiled_k + thread_row, col, ldb, transB) : 0.0f;

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            float a = __shfl_sync(0xFFFFFFFF, A_value, k, BLOCK_SIZE);
            float b = __shfl_sync(0xFFFFFFFF, B_value, k, BLOCK_SIZE);
            C_value += a * b;
        }
    }

    if (row < M && col < N) {
        C[row * ldc + col] = C_value;
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

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_kernel_warp_primitive<<<gridDim, blockDim>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K, lda, ldb, ldc, transA, transB);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication with warp-level primitive optimization (CUDA)");
}