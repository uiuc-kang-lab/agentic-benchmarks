#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 32  // Match block size to warp size

// Using double buffering for shared memory tiles to overlap global loads with computation

__device__ __forceinline__ float load_global(const float* __restrict__ ptr, int idx) {
    return __ldg(ptr + idx);
}

__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K,
    const int lda, const int ldb, const int ldc,
    const bool transA, const bool transB) {
    
    // Shared memory with padding to avoid bank conflicts
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + 1];
    
    // Calculate global indices
    const int by = blockIdx.y * BLOCK_SIZE;
    const int bx = blockIdx.x * BLOCK_SIZE;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    
    // Calculate global row and column
    const int row = by + ty;
    const int col = bx + tx;
    
    // Initialize accumulator
    float sum = 0.0f;
    
    // Calculate number of tiles
    const int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Precompute stride offsets for A and B based on transpose flags
    const int strideA = transA ? 1 : lda;
    const int strideB = transB ? 1 : ldb;
    const int offsetA = transA ? lda : 1;
    const int offsetB = transB ? ldb : 1;
    
    // Loop over tiles
    for (int t = 0; t < numTiles; t++) {
        const int tileK = t * BLOCK_SIZE;
        
        // Load tile into shared memory
        // Use warp-aligned accesses
        if (row < M && (tileK + tx) < K) {
            const int idxA = transA ? (tileK + tx) * lda + row : row * lda + (tileK + tx);
            As[ty][tx] = load_global(A, idxA);
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if ((tileK + ty) < K && col < N) {
            const int idxB = transB ? col * ldb + (tileK + ty) : (tileK + ty) * ldb + col;
            Bs[ty][tx] = load_global(B, idxB);
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot products
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result if within bounds
    // Use a single condition to minimize divergence
    const bool valid = (row < M) && (col < N);
    if (valid) {
        C[row * ldc + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("Input tensors must be on CUDA devices");
    }
    if (A.dim() != 2 || B.dim() != 2) {
        throw std::invalid_argument("Input tensors must be 2D matrices");
    }

    const int64_t A_rows = A.size(0);
    const int64_t A_cols = A.size(1);
    const int64_t B_rows = B.size(0);
    const int64_t B_cols = B.size(1);

    bool transA = false;
    bool transB = false;
    int M, N, K;
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

    // Configure grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb, ldc,
        transA, transB);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Warp-aligned matrix multiplication (CUDA)");
}