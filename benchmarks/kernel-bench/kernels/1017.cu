#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// Tile dimensions and unroll factor
#define TILE_DIM 16
#define UNROLL_FACTOR 4  // Must evenly divide TILE_DIM

// Use __ldg for read-only loads from global memory
__device__ inline float load_elem(const float* __restrict__ matrix, int index) {
    return __ldg(&matrix[index]);
}

// Helper to get an element from matrix A (handles optional transposition)
__device__ inline float getA(const float* A, int row, int col, int ld, bool transA) {
    int idx = transA ? (col * ld + row) : (row * ld + col);
    return load_elem(A, idx);
}

// Helper to get an element from matrix B (handles optional transposition)
__device__ inline float getB(const float* B, int row, int col, int ld, bool transB) {
    int idx = transB ? (col * ld + row) : (row * ld + col);
    return load_elem(B, idx);
}

// Tiled matrix multiplication kernel with shared memory and inner-loop unrolling
__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K,
                                    int lda, int ldb, int ldc,
                                    bool transA, bool transB) {
    // Shared memory tiles for A and B
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    // Compute the row and column index of the C element
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float sum = 0.0f;

    // Loop over tiles along the K dimension
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
        int aCol = t * TILE_DIM + threadIdx.x;
        int bRow = t * TILE_DIM + threadIdx.y;

        // Load tile from A into shared memory
        if (row < M && aCol < K)
            As[threadIdx.y][threadIdx.x] = getA(A, row, aCol, lda, transA);
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile from B into shared memory
        if (bRow < K && col < N)
            Bs[threadIdx.x][threadIdx.y] = getB(B, bRow, col, ldb, transB);
        else
            Bs[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();

        // Unroll the inner loop over the tile dimension
        // Assumption: TILE_DIM is a multiple of UNROLL_FACTOR
        #pragma unroll
        for (int k = 0; k < TILE_DIM; k += UNROLL_FACTOR) {
            sum += As[threadIdx.y][k]     * Bs[k][threadIdx.x];
            sum += As[threadIdx.y][k + 1] * Bs[k + 1][threadIdx.x];
            sum += As[threadIdx.y][k + 2] * Bs[k + 2][threadIdx.x];
            sum += As[threadIdx.y][k + 3] * Bs[k + 3][threadIdx.x];
        }

        __syncthreads();
    }

    // Write back the result
    if (row < M && col < N) {
        C[row * ldc + col] = sum;
    }
}

// Host function: deduces dimensions, handles transposition flags, and launches the kernel
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("Input tensors must be on CUDA devices");
    }
    if (A.dim() != 2 || B.dim() != 2) {
        throw std::invalid_argument("Input tensors must be 2D matrices");
    }

    // Determine matrix sizes and whether a tensor is transposed
    int64_t A_rows = A.size(0);
    int64_t A_cols = A.size(1);
    int64_t B_rows = B.size(0);
    int64_t B_cols = B.size(1);

    bool transA = false;
    bool transB = false;
    int M, N, K;
    int lda, ldb, ldc;
    
    // Four cases to determine layouts
    if (A_rows >= A_cols && B_rows == A_cols) {
        // A: M x K, B: K x N (no transpose)
        M = A_rows;
        K = A_cols;
        N = B_cols;
        lda = A.stride(0);
        ldb = B.stride(0);
    } else if (A_cols > A_rows && B_rows == A_rows) {
        // A stored transposed
        transA = true;
        M = A_cols;
        K = A_rows;
        N = B_cols;
        lda = A.stride(1);
        ldb = B.stride(0);
    } else if (A_rows >= A_cols && B_cols == A_cols) {
        // B stored transposed
        transB = true;
        M = A_rows;
        K = A_cols;
        N = B_rows;
        lda = A.stride(0);
        ldb = B.stride(1);
    } else if (A_cols > A_rows && B_cols == A_rows) {
        // Both A and B stored transposed
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
    
    ldc = N;  // Output assumed to be row-major

    auto C = torch::empty({M, N}, A.options());
    
    // Configure 2D grid and block sizes
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((N + TILE_DIM - 1) / TILE_DIM,
                 (M + TILE_DIM - 1) / TILE_DIM);

    // Launch the kernel
    matmul_tiled_kernel<<<gridDim, blockDim>>>(
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
    m.def("forward", &matmul_cuda, "Tiled Matrix Multiplication with Inner Loop Unrolling (CUDA)");
}
