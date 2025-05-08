#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__device__ __forceinline__ float get_element_ldg(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    if (transpose)
        return __ldg(&matrix[col * ld + row]);
    else
        return __ldg(&matrix[row * ld + col]);
}

__global__ void matmul_kernel_ldg(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K,
                                  int lda, int ldb, int ldc,
                                  bool transA, bool transB) {
    // Block row and column indices
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    // Thread row and column indices within the block
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    // Global row and column indices
    int row = block_row * BLOCK_SIZE + thread_row;
    int col = block_col * BLOCK_SIZE + thread_col;

    float C_value = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load tiles into shared memory ensuring memory coalescing by aligning access
        if (row < M && t * BLOCK_SIZE + thread_col < K)
            As[thread_row][thread_col] = get_element_ldg(A, row, t * BLOCK_SIZE + thread_col, lda, transA);
        else
            As[thread_row][thread_col] = 0.0f;

        if (col < N && t * BLOCK_SIZE + thread_row < K)
            Bs[thread_row][thread_col] = get_element_ldg(B, t * BLOCK_SIZE + thread_row, col, ldb, transB);
        else
            Bs[thread_row][thread_col] = 0.0f;

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            C_value += As[thread_row][k] * Bs[k][thread_col];
        }

        __syncthreads();
    }

    // Write the result to global memory with consideration for alignment
    if (row < M && col < N)
        C[row * ldc + col] = C_value;
}

torch::Tensor matmul_cuda_ldg(torch::Tensor A, torch::Tensor B) {
    // Ensure inputs are CUDA tensors
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("Input tensors must be on CUDA devices");
    }

    // Ensure inputs are 2D matrices
    if (A.dim() != 2 || B.dim() != 2) {
        throw std::invalid_argument("Input tensors must be 2D matrices");
    }

    // Get input dimensions
    int64_t A_rows = A.size(0);
    int64_t A_cols = A.size(1);
    int64_t B_rows = B.size(0);
    int64_t B_cols = B.size(1);

    bool transA = false;
    bool transB = false;
    int64_t M, N, K;
    int lda, ldb, ldc;

    // Determine transpose operations based on shapes
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

    // Allocate output tensor
    auto C = torch::empty({M, N}, A.options());

    // Configure grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the kernel
    matmul_kernel_ldg<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb, ldc,
        transA, transB);

    // Synchronize to ensure completion
    cudaDeviceSynchronize();

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda_ldg, "Matrix multiplication with optimized readonly memory access and alignment (CUDA)");
}