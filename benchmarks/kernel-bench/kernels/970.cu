#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// This kernel uses asynchronous copy (cp.async) to overlap global memory transfers with computation
// in a double-buffering scheme. It supports optional transposition for both input matrices.

__global__ void matmul_kernel_async(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int M, int N, int K,
                                      int lda, int ldb, int ldc,
                                      bool transA, bool transB) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * BLOCK_SIZE + ty;
    const int col = blockIdx.x * BLOCK_SIZE + tx;
    const int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Double-buffered shared memory for tiles of A and B
    __shared__ float A_tile[2][BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_tile[2][BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;

    // Preload first tile (tile 0) into buffer 0 asynchronously
    int tile0 = 0;
    int tileOffset = tile0 * BLOCK_SIZE;
    if (row < M && (tileOffset + tx) < K) {
        const float* srcA = (!transA) ? (A + row * lda + tileOffset + tx) : (A + (tileOffset + tx) * lda + row);
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;"
                     :
                     : "r"(&A_tile[0][ty][tx]), "l"(srcA), "n"(sizeof(float)));
    } else {
        A_tile[0][ty][tx] = 0.0f;
    }
    if (col < N && (tileOffset + ty) < K) {
        const float* srcB = (!transB) ? (B + (tileOffset + ty) * ldb + col) : (B + col * ldb + tileOffset + ty);
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;"
                     :
                     : "r"(&B_tile[0][ty][tx]), "l"(srcB), "n"(sizeof(float)));
    } else {
        B_tile[0][ty][tx] = 0.0f;
    }
    asm volatile("cp.async.commit_group;"); 
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    // Loop over every tile of K, overlapping computation with asynchronous copy
    for (int t = 0; t < numTiles; t++) {
        int cur_buf = t & 1; // current buffer index
        int next_tile = t + 1;
        if (next_tile < numTiles) {
            int next_buf = next_tile & 1; // alternate buffer index
            int nextTileOffset = next_tile * BLOCK_SIZE;
            if (row < M && (nextTileOffset + tx) < K) {
                const float* srcA = (!transA) ? (A + row * lda + nextTileOffset + tx) : (A + (nextTileOffset + tx) * lda + row);
                asm volatile("cp.async.ca.shared.global [%0], [%1], %2;"
                             :
                             : "r"(&A_tile[next_buf][ty][tx]), "l"(srcA), "n"(sizeof(float)));
            } else {
                A_tile[next_buf][ty][tx] = 0.0f;
            }
            if (col < N && (nextTileOffset + ty) < K) {
                const float* srcB = (!transB) ? (B + (nextTileOffset + ty) * ldb + col) : (B + col * ldb + nextTileOffset + ty);
                asm volatile("cp.async.ca.shared.global [%0], [%1], %2;"
                             :
                             : "r"(&B_tile[next_buf][ty][tx]), "l"(srcB), "n"(sizeof(float)));
            } else {
                B_tile[next_buf][ty][tx] = 0.0f;
            }
            asm volatile("cp.async.commit_group;");
        }

        // Ensure the current tile's asynchronous copy is complete
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        // Compute partial product using the current tile
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += A_tile[cur_buf][ty][k] * B_tile[cur_buf][k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * ldc + col] = sum;
    }
}


// Host function for matrix multiplication
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

    // Determine transpose conditions based on input shapes
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

    // Launch kernel using asynchronous copy to overlap memory transfers and computation
    matmul_kernel_async<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb, ldc,
        transA, transB);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication with asynchronous copies to overlap computation and memory transfers (CUDA)");
}
