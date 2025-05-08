#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Experiment with block sizes; here we choose BLOCK_SIZE=32 based on hardware profiling
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

// Helper function to get element with optional transpose
__device__ float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    return transpose ? matrix[col * ld + row] : matrix[row * ld + col];
}

// Kernel using shared memory tiling with block size BLOCK_SIZE x BLOCK_SIZE
__global__ void matmul_kernel(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int M, int N, int K,
                               int lda, int ldb, int ldc,
                               bool transA, bool transB) {
    // Compute block and thread index
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    // Global indices in C
    int row = block_row * BLOCK_SIZE + thread_row;
    int col = block_col * BLOCK_SIZE + thread_col;

    float C_value = 0.0f;
    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Loop over tiles
    for (int t = 0; t < numTiles; t++) {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        int tiledCol = t * BLOCK_SIZE + thread_col;
        int tiledRow = t * BLOCK_SIZE + thread_row;

        // Load A tile
        if (row < M && tiledCol < K)
            As[thread_row][thread_col] = get_element(A, row, tiledCol, lda, transA);
        else
            As[thread_row][thread_col] = 0.0f;

        // Load B tile
        if (col < N && tiledRow < K)
            Bs[thread_row][thread_col] = get_element(B, tiledRow, col, ldb, transB);
        else
            Bs[thread_row][thread_col] = 0.0f;

        __syncthreads();

        // Cache the row of A and column of B in registers
        float A_cache[BLOCK_SIZE];
        float B_cache[BLOCK_SIZE];
        
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            A_cache[k] = As[thread_row][k];
            B_cache[k] = Bs[k][thread_col];
        }
        
        // Compute partial product with cached values
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            C_value += A_cache[k] * B_cache[k];
        }

        __syncthreads();
    }

    // Write the computed value
    if (row < M && col < N)
        C[row * ldc + col] = C_value;
}

// CUDA function interfacing with PyTorch
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Validate input tensors
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("Input tensors must be on CUDA devices");
    }
    if (A.dim() != 2 || B.dim() != 2) {
        throw std::invalid_argument("Input tensors must be 2D matrices");
    }

    // Get dimensions
    int64_t A_rows = A.size(0);
    int64_t A_cols = A.size(1);
    int64_t B_rows = B.size(0);
    int64_t B_cols = B.size(1);

    bool transA = false;
    bool transB = false;
    int64_t M, N, K;
    int lda, ldb, ldc;

    // Determine matrix shapes (and transposition) based on dimensions
    if (A_rows >= A_cols && B_rows == A_cols) {
        // A is (M x K) and B is (K x N)
        M = A_rows;
        K = A_cols;
        N = B_cols;
        lda = A.stride(0);
        ldb = B.stride(0);
    } else if (A_cols > A_rows && B_rows == A_rows) {
        // A is transposed, so treat it as (M x K) with transA true
        transA = true;
        M = A_cols;
        K = A_rows;
        N = B_cols;
        lda = A.stride(1);
        ldb = B.stride(0);
    } else if (A_rows >= A_cols && B_cols == A_cols) {
        // B is transposed
        transB = true;
        M = A_rows;
        K = A_cols;
        N = B_rows;
        lda = A.stride(0);
        ldb = B.stride(1);
    } else if (A_cols > A_rows && B_cols == A_rows) {
        // Both matrices require transposing
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

    // Configure block and grid dimensions based on the new BLOCK_SIZE
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

    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Optimized matrix multiplication with block size experimentation (CUDA)");
}
