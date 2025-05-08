#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// Use a 16x16 tile for shared memory tiling
#define BLOCK_SIZE 32

// Inline function for read-only load using __ldg
__device__ inline float load_elem(const float* __restrict__ matrix, int index) {
    return __ldg(&matrix[index]);
}

// Helpers to fetch elements from A and B respecting transposition
__device__ inline float getA(const float* A, int row, int col, int ld, bool transA) {
    // If transposed, swap row with col
    int idx = transA ? (col * ld + row) : (row * ld + col);
    return load_elem(A, idx);
}

__device__ inline float getB(const float* B, int row, int col, int ld, bool transB) {
    int idx = transB ? (col * ld + row) : (row * ld + col);
    return load_elem(B, idx);
}

// Combined kernel: builds on shared memory tiling (for locality) and inner loop unrolling (for throughput)
// Each block computes a tile of the output matrix C
__global__ void matmul_tiled_unroll_kernel(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            int M, int N, int K,
                                            int lda, int ldb, int ldc,
                                            bool transA, bool transB) {
    // Allocate shared memory for the A and B subtiles
    __shared__ __align__(16) float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ __align__(16) float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Compute the global row and column indices for C
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Loop over tiles along the K dimension
    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * BLOCK_SIZE + threadIdx.x;
        int bRow = t * BLOCK_SIZE + threadIdx.y;

        // Load tile from matrix A, with bounds check
        if (row < M && aCol < K)
            As[threadIdx.y][threadIdx.x] = getA(A, row, aCol, lda, transA);
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile from matrix B, with bounds check
        if (bRow < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = getB(B, bRow, col, ldb, transB);
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Unroll the inner loop by factor of 4 to compute partial products
        // BLOCK_SIZE is chosen as 16 (divisible by 4), so this is safe
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k += 4) {
            sum += As[threadIdx.y][k]     * Bs[k][threadIdx.x]     +
                   As[threadIdx.y][k + 1] * Bs[k + 1][threadIdx.x] +
                   As[threadIdx.y][k + 2] * Bs[k + 2][threadIdx.x] +
                   As[threadIdx.y][k + 3] * Bs[k + 3][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the computed value back to the output matrix if within bounds
    if (row < M && col < N) {
        C[row * ldc + col] = sum;
    }
}

// Host function interfacing with PyTorch
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("Input tensors must be on CUDA devices");
    }
    if (A.dim() != 2 || B.dim() != 2) {
        throw std::invalid_argument("Input tensors must be 2D matrices");
    }

    // Retrieve dimensions and decide on possible transposition logic
    int64_t A_rows = A.size(0);
    int64_t A_cols = A.size(1);
    int64_t B_rows = B.size(0);
    int64_t B_cols = B.size(1);

    bool transA = false;
    bool transB = false;
    int M, N, K;
    int lda, ldb, ldc;

    if (A_rows >= A_cols && B_rows == A_cols) {
        // A is MxK, B is KxN
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
    ldc = N;

    auto C = torch::empty({M, N}, A.options());

    // Configure grid and block dimensions
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the kernel
    matmul_tiled_unroll_kernel<<<grid, block>>>(
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
    m.def("forward", &matmul_cuda, "Tiled Matrix Multiplication with loop unrolling (CUDA)");
}
