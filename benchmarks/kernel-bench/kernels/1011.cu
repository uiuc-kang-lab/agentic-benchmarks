// Includes for CUDA and PyTorch extensions
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define BLOCK_SIZE 16

// Inline function for read-only load using __ldg
__device__ inline float load_ldg(const float* __restrict__ ptr) {
    return __ldg(ptr);
}

// Unified function to fetch an element from a matrix with optional transposition
__device__ inline float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    int idx = transpose ? (col * ld + row) : (row * ld + col);
    return load_ldg(&matrix[idx]);
}

// Optimized tiled matrix multiplication kernel using __ldg and shared memory
__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K,
                              int lda, int ldb, int ldc,
                              bool transA, bool transB) {
    // Compute global row and column indices
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Shared memory tiles for A and B with 128-bit alignment
    __shared__ __align__(16) float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ __align__(16) float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Loop over tiles along the K dimension
    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * BLOCK_SIZE + threadIdx.x; // Column index for A
        int bRow = t * BLOCK_SIZE + threadIdx.y; // Row index for B

        // Load tile of A into shared memory using __ldg
        if (row < M && aCol < K) {
            As[threadIdx.y][threadIdx.x] = get_element(A, row, aCol, lda, transA);
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile of B into shared memory using __ldg
        if (bRow < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = get_element(B, bRow, col, ldb, transB);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial product from the tile
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result to global memory if within bounds
    if (row < M && col < N) {
        C[row * ldc + col] = sum;
    }
}

// Host function to setup and launch the CUDA kernel
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Check that input tensors are CUDA tensors
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("Input tensors must be on CUDA devices");
    }
    
    // Ensure input tensors are 2D matrices
    if (A.dim() != 2 || B.dim() != 2) {
        throw std::invalid_argument("Input tensors must be 2D matrices");
    }
    
    // Get dimensions of A and B
    int64_t A_rows = A.size(0);
    int64_t A_cols = A.size(1);
    int64_t B_rows = B.size(0);
    int64_t B_cols = B.size(1);

    bool transA = false;
    bool transB = false;
    int M, N, K;
    int lda, ldb, ldc;

    // Determine configuration based on matrix shapes
    if (A_rows >= A_cols && B_rows == A_cols) {
        // Case: A (M x K), B (K x N)
        M = A_rows;
        K = A_cols;
        N = B_cols;
        lda = A.stride(0);
        ldb = B.stride(0);
    } else if (A_cols > A_rows && B_rows == A_rows) {
        // Case: A stored transposed (K x M)
        transA = true;
        M = A_cols;
        K = A_rows;
        N = B_cols;
        lda = A.stride(1);
        ldb = B.stride(0);
    } else if (A_rows >= A_cols && B_cols == A_cols) {
        // Case: B stored transposed (N x K)
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
    
    ldc = N;  // Leading dimension for output matrix C

    auto C = torch::empty({M, N}, A.options());

    // Configure grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the CUDA kernel
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

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Optimized Tiled Matrix Multiplication with __ldg and unified transposition (CUDA)");
}
