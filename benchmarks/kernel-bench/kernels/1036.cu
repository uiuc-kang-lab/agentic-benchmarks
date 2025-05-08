#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Use a larger block size to help coalescing on modern GPUs
#define BLOCK_SIZE 32

// -----------------------------------------------------------------------------
// Specialized kernel for the common (non-transposed) case: A is MxK, B is KxN
// with both matrices stored in row-major order. Global memory loads are
// arranged so that consecutive threads in a warp access contiguous memory.
// -----------------------------------------------------------------------------
__global__ void matmul_kernel_coalesced(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int M, int N, int K,
                                          int lda, int ldb, int ldc) {
    // Compute global row and column for C
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float sum = 0.0f;

    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    // Loop over tiles
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        int aCol = t * BLOCK_SIZE + threadIdx.x;  // consecutive in memory for A
        int bRow = t * BLOCK_SIZE + threadIdx.y;    // consecutive in memory for B

        // Load A tile: each thread loads one element
        if (row < M && aCol < K)
            sA[threadIdx.y][threadIdx.x] = A[row * lda + aCol];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B tile: each thread loads one element
        if (bRow < K && col < N)
            sB[threadIdx.y][threadIdx.x] = B[bRow * ldb + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Multiply the two tiles
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    // Write the result to global memory
    if (row < M && col < N)
        C[row * ldc + col] = sum;
}

// -----------------------------------------------------------------------------
// General kernel that supports transposed cases using a helper function.
// It sacrifices some coalescing but guarantees correctness.
// -----------------------------------------------------------------------------
__device__ float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    return transpose ? matrix[col * ld + row] : matrix[row * ld + col];
}

__global__ void matmul_kernel_general(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int M, int N, int K,
                                        int lda, int ldb, int ldc,
                                        bool transA, bool transB) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float sum = 0.0f;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        int aRow = row;
        int aCol = t * BLOCK_SIZE + threadIdx.x;
        if (aRow < M && aCol < K)
            As[threadIdx.y][threadIdx.x] = get_element(A, aRow, aCol, lda, transA);
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        int bRow = t * BLOCK_SIZE + threadIdx.y;
        int bCol = col;
        if (bRow < K && bCol < N)
            Bs[threadIdx.y][threadIdx.x] = get_element(B, bRow, bCol, ldb, transB);
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * ldc + col] = sum;
}

// -----------------------------------------------------------------------------
// Host function: determines matrix shapes and selects the optimized kernel
// when possible. If both matrices are non-transposed (typical row-major),
// the coalesced kernel is launched. Otherwise the general kernel is used.
// -----------------------------------------------------------------------------

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Check that inputs are CUDA tensors and 2D
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("Input tensors must be on CUDA devices");
    }
    if (A.dim() != 2 || B.dim() != 2) {
        throw std::invalid_argument("Input tensors must be 2D matrices");
    }

    // Determine dimensions and whether matrices are to be transposed
    int64_t A_rows = A.size(0);
    int64_t A_cols = A.size(1);
    int64_t B_rows = B.size(0);
    int64_t B_cols = B.size(1);

    bool transA = false;
    bool transB = false;
    int64_t M, N, K;
    int lda, ldb, ldc;

    // Four cases based on the matrix shapes
    if (A_rows >= A_cols && B_rows == A_cols) {
        // A: (M x K), B: (K x N)
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

    // If both matrices are non-transposed, use the coalesced kernel
    if (!transA && !transB) {
        matmul_kernel_coalesced<<<gridDim, blockDim>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, N, K,
            lda, ldb, ldc);
    } else {
        matmul_kernel_general<<<gridDim, blockDim>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, N, K,
            lda, ldb, ldc,
            transA, transB);
    }

    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Optimized matrix multiplication with memory coalescing (CUDA)");
}
