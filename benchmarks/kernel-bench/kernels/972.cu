#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// This device function handles element fetching with an option to use transposed indexing.
__device__ float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    return transpose ? matrix[col * ld + row] : matrix[row * ld + col];
}

// CUDA kernel that minimizes warp divergence by checking for full-tile conditions
// and refactoring conditional logic to ensure uniform control flow across a warp.
__global__ void matmul_kernel_uniform(const float* __restrict__ A,
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

    // Loop over tiles
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        int tiled_k = t * BLOCK_SIZE;

        // Determine if the current tile is fully inside matrix boundaries for both A and B.
        bool full_tile_A = ((block_row * BLOCK_SIZE + BLOCK_SIZE) <= M) && ((tiled_k + BLOCK_SIZE) <= K);
        bool full_tile_B = ((block_col * BLOCK_SIZE + BLOCK_SIZE) <= N) && ((tiled_k + BLOCK_SIZE) <= K);
        bool full_tile = full_tile_A && full_tile_B;

        if (full_tile) {
            // When the tile is fully within bounds, all threads follow the same path.
            As[thread_row][thread_col] = get_element(A, row, tiled_k + thread_col, lda, transA);
            Bs[thread_row][thread_col] = get_element(B, tiled_k + thread_row, col, ldb, transB);
        } else {
            // For boundary tiles, ensure out-of-bound accesses return 0.0f
            As[thread_row][thread_col] = (row < M && (tiled_k + thread_col) < K) ? 
                                           get_element(A, row, tiled_k + thread_col, lda, transA) : 0.0f;
            Bs[thread_row][thread_col] = (col < N && (tiled_k + thread_row) < K) ? 
                                           get_element(B, tiled_k + thread_row, col, ldb, transB) : 0.0f;
        }

        __syncthreads();

        // Compute partial sum for this tile with unrolled inner loop
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            C_value += As[thread_row][k] * Bs[k][thread_col];
        }

        __syncthreads();
    }

    // Write the output if within bounds
    if (row < M && col < N) {
        C[row * ldc + col] = C_value;
    }
}

// Host function to prepare and launch the CUDA kernel
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

    // Determine matrix dimensions and transpose flags
    if (A_rows >= A_cols && B_rows == A_cols) {
        // A: M x K, B: K x N
        M = A_rows;
        K = A_cols;
        N = B_cols;
        lda = A.stride(0);
        ldb = B.stride(0);
    } else if (A_cols > A_rows && B_rows == A_rows) {
        // A is transposed: effective A: M x K
        transA = true;
        M = A_cols;
        K = A_rows;
        N = B_cols;
        lda = A.stride(1);
        ldb = B.stride(0);
    } else if (A_rows >= A_cols && B_cols == A_cols) {
        // B is transposed: effective B: K x N
        transB = true;
        M = A_rows;
        K = A_cols;
        N = B_rows;
        lda = A.stride(0);
        ldb = B.stride(1);
    } else if (A_cols > A_rows && B_cols == A_rows) {
        // Both A and B are transposed
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

    // Allocate the output tensor
    auto C = torch::empty({M, N}, A.options());

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_kernel_uniform<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb, ldc,
        transA, transB);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication with uniform control flow to minimize warp divergence (CUDA)");
}
