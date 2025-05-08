#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Use a larger tile size and add padding in shared memory to avoid bank conflicts
#define TILE_SIZE 32

// Device helper function: obtain matrix element with optional transpose
__device__ inline float get_element(const float* __restrict__ matrix,
                                      int row, int col, int ld, bool transpose) {
    return transpose ? matrix[col * ld + row] : matrix[row * ld + col];
}

// Optimized tiled matrix multiplication kernel
__global__ void matmul_opt_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K,
                                    int lda, int ldb, int ldc,
                                    bool transA, bool transB) {
    // Allocate shared memory with padding to reduce bank conflicts
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Compute the row and column index of the C element to work on
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over the tiles of the K dimension
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int tiled_k = t * TILE_SIZE;

        // Load one tile of A into shared memory in a coalesced way
        if (row < M && (tiled_k + tx) < K) {
            As[ty][tx] = get_element(A, row, tiled_k + tx, lda, transA);
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load one tile of B into shared memory in a coalesced way
        if ((tiled_k + ty) < K && col < N) {
            Bs[ty][tx] = get_element(B, tiled_k + ty, col, ldb, transB);
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Perform computation on the tile with unrolling for optimization
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }

    // Write the computed value to global memory if within bounds
    if (row < M && col < N) {
        C[row * ldc + col] = sum;
    }
}

// Host function callable from PyTorch
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("Input tensors must be on CUDA devices");
    }

    if (A.dim() != 2 || B.dim() != 2) {
        throw std::invalid_argument("Input tensors must be 2D matrices");
    }

    // Determine dimensions and whether we need to transpose A or B
    int64_t A_rows = A.size(0);
    int64_t A_cols = A.size(1);
    int64_t B_rows = B.size(0);
    int64_t B_cols = B.size(1);

    bool transA = false;
    bool transB = false;
    int64_t M, N, K;
    int lda, ldb, ldc;

    // Figure out the multiplication logic based on input shapes
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

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_opt_kernel<<<grid, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb, ldc,
        transA, transB);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Optimized Matrix Multiplication with Tiled Shared Memory (CUDA)");
}
