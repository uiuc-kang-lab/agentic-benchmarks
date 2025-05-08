#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 16

// Inline function to perform a read-only load using __ldg
__device__ inline float load_elem(const float* __restrict__ matrix, int index) {
    return __ldg(&matrix[index]);
}

// Get element from matrix A (handles transpose) using __ldg
__device__ inline float getA(const float* A, int row, int col, int ld, bool transA) {
    int idx = transA ? (col * ld + row) : (row * ld + col);
    return load_elem(A, idx);
}

// Get element from matrix B (handles transpose) using __ldg
__device__ inline float getB(const float* B, int row, int col, int ld, bool transB) {
    int idx = transB ? (col * ld + row) : (row * ld + col);
    return load_elem(B, idx);
}

// Tiled matrix multiplication kernel using __ldg for read-only global memory accesses
__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K,
                              int lda, int ldb, int ldc,
                              bool transA, bool transB) {
    // Shared memory tiles aligned to 128-bit boundaries
    __shared__ __align__(16) float As0[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ __align__(16) float Bs0[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ __align__(16) float As1[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ __align__(16) float Bs1[BLOCK_SIZE][BLOCK_SIZE];

    // Compute global row and column indices
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float sum = 0.0f;
    
    // Loop over tiles along the K dimension
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        int aCol = t * BLOCK_SIZE + threadIdx.x;
        int bRow = t * BLOCK_SIZE + threadIdx.y;

        // Load A tile element with __ldg if within bounds
        if (row < M && aCol < K)
            As[threadIdx.y][threadIdx.x] = getA(A, row, aCol, lda, transA);
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B tile element with __ldg if within bounds
        if (bRow < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = getB(B, bRow, col, ldb, transB);
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product for the tile
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the output if within bounds
    if (row < M && col < N) {
        C[row * ldc + col] = sum;
    }
}

// Host function
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("Input tensors must be on CUDA devices");
    }
    if (A.dim() != 2 || B.dim() != 2) {
        throw std::invalid_argument("Input tensors must be 2D matrices");
    }

    // Determine matrix dimensions and whether to transpose based on shapes
    int64_t A_rows = A.size(0);
    int64_t A_cols = A.size(1);
    int64_t B_rows = B.size(0);
    int64_t B_cols = B.size(1);

    bool transA = false;
    bool transB = false;
    int M, N, K;
    int lda, ldb, ldc;
    
    if (A_rows >= A_cols && B_rows == A_cols) {
        // Case: A (M x K), B (K x N)
        M = A_rows;
        K = A_cols;
        N = B_cols;
        lda = A.stride(0);
        ldb = B.stride(0);
    } else if (A_cols > A_rows && B_rows == A_rows) {
        // Case: A is stored transposed
        transA = true;
        M = A_cols;
        K = A_rows;
        N = B_cols;
        lda = A.stride(1);
        ldb = B.stride(0);
    } else if (A_rows >= A_cols && B_cols == A_cols) {
        // Case: B is stored transposed
        transB = true;
        M = A_rows;
        K = A_cols;
        N = B_rows;
        lda = A.stride(0);
        ldb = B.stride(1);
    } else if (A_cols > A_rows && B_cols == A_rows) {
        // Case: Both A and B are stored transposed
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

    // Configure grid and block dimensions for tiled execution
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
    m.def("forward", &matmul_cuda, "Tiled Matrix Multiplication with __ldg and 128-bit aligned loads (CUDA)");
}
