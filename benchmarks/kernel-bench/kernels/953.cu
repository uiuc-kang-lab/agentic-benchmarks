#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 32 // Same as WARP_SIZE for efficiency

__device__ float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    if (transpose)
        return matrix[col * ld + row];
    else
        return matrix[row * ld + col];
}

__global__ void matmul_warp_intrinsics_kernel(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              int M, int N, int K,
                                              int lda, int ldb, int ldc,
                                              bool transA, bool transB) {
    // Each block processes a tile of C
    int blockRow = blockIdx.y * BLOCK_SIZE;
    int blockCol = blockIdx.x * BLOCK_SIZE;

    // Initialize C value for this thread
    float CValue = 0.0f;

    // Loop over the elements of A and B required to compute the block result
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Calculate the row and column index
        int row = blockRow + threadIdx.y;
        int col = blockCol + threadIdx.x;

        // Load elements of A and B
        float AValue = 0.0f;
        if (row < M && (t * BLOCK_SIZE + threadIdx.x) < K) {
            AValue = get_element(A, row, t * BLOCK_SIZE + threadIdx.x, lda, transA);
        }
        float BValue = 0.0f;
        if ((t * BLOCK_SIZE + threadIdx.y) < K && col < N) {
            BValue = get_element(B, t * BLOCK_SIZE + threadIdx.y, col, ldb, transB);
        }

        // Use warp shuffle intrinsics to perform reductions across the warp
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            float AV = __shfl_sync(0xFFFFFFFF, AValue, k);
            float BV = __shfl_sync(0xFFFFFFFF, BValue, k);
            CValue += AV * BV;
        }
    }

    // Store the result in C
    int row = blockRow + threadIdx.y;
    int col = blockCol + threadIdx.x;
    if (row < M && col < N) {
        C[row * ldc + col] = CValue;
    }
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

    matmul_warp_intrinsics_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb, ldc,
        transA, transB);

    // Remove unnecessary cudaDeviceSynchronize to rely on PyTorch synchronization
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication using warp-level intrinsics (CUDA)");
}