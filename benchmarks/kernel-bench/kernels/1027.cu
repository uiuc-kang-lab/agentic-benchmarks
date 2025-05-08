#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__device__ float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    if (transpose)
        return matrix[col * ld + row];
    else
        return matrix[row * ld + col];
}

__global__ void optimized_matmul_kernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int M, int N, int K,
                                         int lda, int ldb, int ldc,
                                         bool transA, bool transB) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;

    float C_value = 0.0f;

    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        int tiledRow = blockRow * TILE_WIDTH + row;
        int tiledCol = blockCol * TILE_WIDTH + col;
        
        if (tiledRow < M && t * TILE_WIDTH + col < K)
            As[row][col] = get_element(A, tiledRow, t * TILE_WIDTH + col, lda, transA);
        else
            As[row][col] = 0.0f;

        if (tiledCol < N && t * TILE_WIDTH + row < K)
            Bs[row][col] = get_element(B, t * TILE_WIDTH + row, tiledCol, ldb, transB);
        else
            Bs[row][col] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            C_value += As[row][k] * Bs[k][col];
        }

        __syncthreads();
    }

    int globalRow = blockRow * TILE_WIDTH + row;
    int globalCol = blockCol * TILE_WIDTH + col;
    if (globalRow < M && globalCol < N)
        C[globalRow * ldc + globalCol] = C_value;
}

torch::Tensor matmul_cuda_optimized(torch::Tensor A, torch::Tensor B) {
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

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH,
                 (M + TILE_WIDTH - 1) / TILE_WIDTH);

    optimized_matmul_kernel<<<gridDim, blockDim>>>(
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
    m.def("forward", &matmul_cuda_optimized, "Optimized Matrix multiplication with efficient thread and block indexing (CUDA)");
}