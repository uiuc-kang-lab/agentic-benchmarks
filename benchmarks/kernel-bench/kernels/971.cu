#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__constant__ float constA[BLOCK_SIZE * BLOCK_SIZE];
__constant__ float constB[BLOCK_SIZE * BLOCK_SIZE];

__device__ float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    if (transpose)
        return matrix[col * ld + row];
    else
        return matrix[row * ld + col];
}

__global__ void matmul_kernel_constant(const float* __restrict__ A,
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
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        int tiled_k = t * BLOCK_SIZE;
        bool valid_A = (row < M && tiled_k + thread_col < K);
        bool valid_B = (col < N && tiled_k + thread_row < K);

        // Use constant memory for the current tile if possible
        float* A_tile = (float*)(valid_A ? &constA[thread_row * BLOCK_SIZE + thread_col] : 0);
        float* B_tile = (float*)(valid_B ? &constB[thread_row * BLOCK_SIZE + thread_col] : 0);

        As[thread_row][thread_col] = valid_A ? *A_tile : 0.0f;
        Bs[thread_row][thread_col] = valid_B ? *B_tile : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            C_value += As[thread_row][k] * Bs[k][thread_col];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * ldc + col] = C_value;
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("Input tensors must be on CUDA devices");
    }

    if (A.dim() != 2 || B.dim() != 2) {
        throw std::invalid_argument("Input tensors must be 2D matrices");
    }

    int64_t A_rows = A.size(0);
    int64_t A_cols = A.size(1);
    int64_t B_rows = B.size(0);
    int64_t B_cols = B.size(1);

    bool transA = false;
    bool transB = false;
    int64_t M, N, K;
    int lda, ldb, ldc;

    if (A_rows >= A_cols && B_rows == A_cols) {
        M = A_rows;
        K = A_cols;
        N = B_cols;
        lda = A.stride(0);
        ldb = B.stride(0);
    } else if (A_cols > A_rows && B_rows == A_rows) {
        transA = true;
        M = A_cols;
        K = A_rows;
        N = B_cols;
        lda = A.stride(1);
        ldb = B.stride(0);
    } else if (A_rows >= A_cols && B_cols == A_cols) {
        transB = true;
        M = A_rows;
        K = A_cols;
        N = B_rows;
        lda = A.stride(0);
        ldb = B.stride(1);
    } else if (A_cols > A_rows && B_cols == A_rows) {
        transA = true;
        transB = true;
        M = A_cols;
        K = A_rows;
        N = B_rows;
        lda = A.stride(1);
        ldb = B.stride(1);
    } else {
        throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
    }

    ldc = N;

    auto C = torch::empty({M, N}, A.options());

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaMemcpyToSymbol(constA, A.data_ptr<float>(), BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
    cudaMemcpyToSymbol(constB, B.data_ptr<float>(), BLOCK_SIZE * BLOCK_SIZE * sizeof(float));

    matmul_kernel_constant<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb, ldc,
        transA, transB);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication with constant memory optimization (CUDA)");
}
