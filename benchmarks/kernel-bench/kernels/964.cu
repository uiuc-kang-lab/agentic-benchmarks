#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// Device function to load an element using __ldg while assuming 128-bit aligned accesses
__device__ __forceinline__ float get_element_ldg(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    // Compute the linear index based on whether the matrix is transposed
    int index = transpose ? col * ld + row : row * ld + col;
    // __ldg performs a read-only, potentially 128-bit wide, load
    return __ldg(&matrix[index]);
}

// Kernel implementing tiled matrix multiplication with __ldg() read-only loads
// Shared memory is declared with an alignment qualifier to help ensure 128-bit alignment
__global__ void matmul_kernel_aligned_ldg(const float* __restrict__ A,
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

    __shared__ __align__(16) float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ __align__(16) float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int t = 0; t < numTiles; t++) {
        int tiled_k = t * BLOCK_SIZE;
        if (row < M && (tiled_k + thread_col) < K)
            As[thread_row][thread_col] = get_element_ldg(A, row, tiled_k + thread_col, lda, transA);
        else
            As[thread_row][thread_col] = 0.0f;

        if (col < N && (tiled_k + thread_row) < K)
            Bs[thread_row][thread_col] = get_element_ldg(B, tiled_k + thread_row, col, ldb, transB);
        else
            Bs[thread_row][thread_col] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            C_value += As[thread_row][k] * Bs[k][thread_col];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * ldc + col] = C_value;
    }
}

// Host function for PyTorch interface
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

    // Determine matrix dimensions and potential transposition based on shapes
    if (A_rows >= A_cols && B_rows == A_cols) {
        // A is MxK, B is KxN
        M = A_rows;
        K = A_cols;
        N = B_cols;
        lda = A.stride(0);
        ldb = B.stride(0);
    } else if (A_cols > A_rows && B_rows == A_rows) {
        // A is stored transposed
        transA = true;
        M = A_cols;
        K = A_rows;
        N = B_cols;
        lda = A.stride(1);
        ldb = B.stride(0);
    } else if (A_rows >= A_cols && B_cols == A_cols) {
        // B is stored transposed
        transB = true;
        M = A_rows;
        K = A_cols;
        N = B_rows;
        lda = A.stride(0);
        ldb = B.stride(1);
    } else if (A_cols > A_rows && B_cols == A_rows) {
        // Both A and B are stored transposed
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

    matmul_kernel_aligned_ldg<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb, ldc,
        transA, transB);

    cudaDeviceSynchronize();

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication using __ldg() with 128-bit aligned loads (CUDA)");
}
