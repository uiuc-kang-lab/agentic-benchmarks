#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// Device function to get elements from A and B considering transpose
__device__ float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    if (transpose)
        return matrix[col * ld + row];
    else
        return matrix[row * ld + col];
}

// Device function to load tiles into shared memory
__device__ void load_tiles(const float* __restrict__ A, const float* __restrict__ B,
                           float As[BLOCK_SIZE][BLOCK_SIZE], float Bs[BLOCK_SIZE][BLOCK_SIZE],
                           int row, int col, int t, int M, int N, int K, int lda, int ldb,
                           bool transA, bool transB) {
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    if (row < M && t * BLOCK_SIZE + thread_col < K)
        As[thread_row][thread_col] = get_element(A, row, t * BLOCK_SIZE + thread_col, lda, transA);
    else
        As[thread_row][thread_col] = 0.0f;

    if (col < N && t * BLOCK_SIZE + thread_row < K)
        Bs[thread_row][thread_col] = get_element(B, t * BLOCK_SIZE + thread_row, col, ldb, transB);
    else
        Bs[thread_row][thread_col] = 0.0f;
}

// Device function to compute partial product
__device__ float compute_partial_product(float As[BLOCK_SIZE][BLOCK_SIZE], float Bs[BLOCK_SIZE][BLOCK_SIZE]) {
    float C_value = 0.0f;
    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        C_value += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    return C_value;
}

__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K,
                              int lda, int ldb, int ldc,
                              bool transA, bool transB) {
    // Block row and column indices
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    // Global row and column indices
    int row = block_row * BLOCK_SIZE + threadIdx.y;
    int col = block_col * BLOCK_SIZE + threadIdx.x;

    float C_value = 0.0f;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    // Loop over tiles
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Load tiles into shared memory
        load_tiles(A, B, As, Bs, row, col, t, M, N, K, lda, ldb, transA, transB);

        __syncthreads();

        // Compute partial product
        C_value += compute_partial_product(As, Bs);

        __syncthreads();
    }

    // Write the result to global memory
    if (row < M && col < N)
        C[row * ldc + col] = C_value;
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
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
        // Case: A (M x K), B (K x N)
        M = A_rows;
        K = A_cols;
        N = B_cols;
        lda = A.stride(0);
        ldb = B.stride(0);
    } else if (A_cols > A_rows && B_rows == A_rows) {
        // Case: A (K x M), transposed
        transA = true;
        M = A_cols;
        K = A_rows;
        N = B_cols;
        lda = A.stride(1);
        ldb = B.stride(0);
    } else if (A_rows >= A_cols && B_cols == A_cols) {
        // Case: B (N x K), transposed
        transB = true;
        M = A_rows;
        K = A_cols;
        N = B_rows;
        lda = A.stride(0);
        ldb = B.stride(1);
    } else if (A_cols > A_rows && B_cols == A_rows) {
        // Case: Both A and B are transposed
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
    matmul_kernel<<<gridDim, blockDim>>>(
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
    m.def("forward", &matmul_cuda, "Matrix multiplication with unrolled loop optimization (CUDA)");
}