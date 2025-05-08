#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// Helper macro
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// Helper function to fetch matrix element; supports transposition
__device__ inline float get_elem(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    return transpose ? matrix[col * ld + row] : matrix[row * ld + col];
}

// Kernel implementing split-K GEMM with minimal atomic operations.
// Each block computes a partial sum for a BLOCK_SIZE x BLOCK_SIZE tile of C over a partition of the K dimension.
// When split_k > 1, multiple blocks collaborate on the same C tile and use atomicAdd to accumulate results.
// If split_k equals 1, each block writes its result directly (no atomics needed).
__global__ void splitk_atomic_matmul_kernel(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              int M, int N, int K,
                                              int lda, int ldb, int ldc,
                                              bool transA, bool transB,
                                              int split_k) {
    // Determine the output tile indices
    int tile_row = blockIdx.y * BLOCK_SIZE;
    int tile_col = blockIdx.x * BLOCK_SIZE;

    // Global row and column computed by this thread
    int row = tile_row + threadIdx.y;
    int col = tile_col + threadIdx.x;

    // Partition K dimension among blocks in the z-dimension
    int chunk_size = (K + split_k - 1) / split_k;
    int k_start = blockIdx.z * chunk_size;
    int k_end = MIN(k_start + chunk_size, K);

    // Accumulator register
    float sum = 0.0f;

    // Shared memory tiles for A and B
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Loop over tiles in the current K partition
    for (int t = k_start; t < k_end; t += BLOCK_SIZE) {
        int current_tile = MIN(BLOCK_SIZE, k_end - t);

        // Load A tile into shared memory
        if (row < M && (t + threadIdx.x) < k_end) {
            // If not transposed, A is M x K; else A is transposed
            As[threadIdx.y][threadIdx.x] = get_elem(A, row, t + threadIdx.x, lda, transA);
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile into shared memory
        if (col < N && (t + threadIdx.y) < k_end) {
            Bs[threadIdx.y][threadIdx.x] = get_elem(B, t + threadIdx.y, col, ldb, transB);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Multiply the tiles
        for (int k_inner = 0; k_inner < current_tile; ++k_inner) {
            sum += As[threadIdx.y][k_inner] * Bs[k_inner][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result to global memory: use atomicAdd if multiple blocks accumulate into the same C element
    if (row < M && col < N) {
        if (split_k > 1) {
            atomicAdd(&C[row * ldc + col], sum);
        } else {
            C[row * ldc + col] = sum;
        }
    }
}

// Host function for matrix multiplication with an optional split_k parameter.
// If split_k > 1, the K dimension is partitioned and C must be pre-initialized to 0.
// When split_k == 1, the kernel writes results directly without atomics.

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B, int split_k=1) {
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

    // Determine matrix shapes and transposition based on dimensions
    if (A_rows >= A_cols && B_rows == A_cols) {
        // A: M x K, B: K x N
        M = A_rows;
        K = A_cols;
        N = B_cols;
        lda = A.stride(0);
        ldb = B.stride(0);
    } else if (A_cols > A_rows && B_rows == A_rows) {
        // A is transposed: A^T is M x K
        transA = true;
        M = A_cols;
        K = A_rows;
        N = B_cols;
        lda = A.stride(1);
        ldb = B.stride(0);
    } else if (A_rows >= A_cols && B_cols == A_cols) {
        // B is transposed: B^T is K x N
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

    // Allocate output tensor C. For split_k > 1, initialize to zeros so that atomic adds work correctly.
    torch::Tensor C = (split_k > 1) ? torch::zeros({M, N}, A.options()) : torch::empty({M, N}, A.options());

    // Configure block and grid dimensions.
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 split_k);

    splitk_atomic_matmul_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb, ldc,
        transA, transB,
        split_k);

    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication with split-K and minimal atomic operations (CUDA)");
}
