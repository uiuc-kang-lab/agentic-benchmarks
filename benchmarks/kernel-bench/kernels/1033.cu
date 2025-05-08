#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define TILE_K 16

// Helper device function to access matrix elements with optional transpose
__forceinline__ __device__ float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    return transpose ? matrix[col * ld + row] : matrix[row * ld + col];
}

// Kernel that splits the K dimension among grid.z blocks and accumulates partial results
__global__ void matmul_atomic_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int N, int K,
                                       int lda, int ldb, int ldc,
                                       bool transA, bool transB) {
    // Compute block indices for output tile and K chunk
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int block_k = blockIdx.z;

    // Global row and column for C
    int row = block_row * BLOCK_SIZE + threadIdx.y;
    int col = block_col * BLOCK_SIZE + threadIdx.x;

    // Determine the K-range this block is responsible for
    int k_start = block_k * TILE_K;
    int k_end = k_start + TILE_K;
    if (k_end > K) k_end = K;

    float sum = 0.0f;

    // Shared memory tiles for A and B for this K-chunk
    __shared__ float As[BLOCK_SIZE][TILE_K];
    __shared__ float Bs[TILE_K][BLOCK_SIZE];

    // Each thread loads one element from A and B into shared memory
    // Loop over the TILE_K dimension. Note that for threads outside of valid indices, zeros are loaded.
    for (int t = 0; t < TILE_K; t++) {
        int k_index = k_start + t;
        if (row < M && k_index < K)
            As[threadIdx.y][t] = get_element(A, row, k_index, lda, transA);
        else
            As[threadIdx.y][t] = 0.0f;

        if (k_index < K && col < N)
            Bs[t][threadIdx.x] = get_element(B, k_index, col, ldb, transB);
        else
            Bs[t][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Perform the computation for the given K-chunk
    if (row < M && col < N) {
        // Only iterate over the valid portion of the tile
        for (int t = 0; t < (k_end - k_start); t++) {
            sum += As[threadIdx.y][t] * Bs[t][threadIdx.x];
        }
    }
    __syncthreads();

    // Accumulate partial sums into global memory
    // If the grid is not split along K (i.e., gridDim.z == 1), we can write directly to C;
    // otherwise, use atomicAdd to safely accumulate contributions from multiple blocks.
    if (row < M && col < N) {
        if (gridDim.z == 1) {
            C[row * ldc + col] = sum;
        } else {
            atomicAdd(&C[row * ldc + col], sum);
        }
    }
}

// Host function that prepares and launches the kernel
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Validate input tensors
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("Input tensors must be on CUDA devices");
    }
    if (A.dim() != 2 || B.dim() != 2) {
        throw std::invalid_argument("Input tensors must be 2D matrices");
    }

    // Get dimensions and determine transpose operations
    int64_t A_rows = A.size(0);
    int64_t A_cols = A.size(1);
    int64_t B_rows = B.size(0);
    int64_t B_cols = B.size(1);

    bool transA = false;
    bool transB = false;
    int64_t M, N, K;
    int lda, ldb, ldc;

    if (A_rows >= A_cols && B_rows == A_cols) {
        // A is (M x K), B is (K x N)
        M = A_rows;
        K = A_cols;
        N = B_cols;
        lda = A.stride(0);
        ldb = B.stride(0);
    } else if (A_cols > A_rows && B_rows == A_rows) {
        // A is (K x M) so it is transposed
        transA = true;
        M = A_cols;
        K = A_rows;
        N = B_cols;
        lda = A.stride(1);
        ldb = B.stride(0);
    } else if (A_rows >= A_cols && B_cols == A_cols) {
        // B is (N x K) so it is transposed
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

    // Allocate output tensor and initialize to zero for safe atomic accumulation
    auto C = torch::zeros({M, N}, A.options());

    // Configure the block and grid dimensions.
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (K + TILE_K - 1) / TILE_K);

    // Launch the kernel
    matmul_atomic_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb, ldc,
        transA, transB
    );

    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Tall skinny matrix multiplication with atomic accumulation (CUDA)");
}
