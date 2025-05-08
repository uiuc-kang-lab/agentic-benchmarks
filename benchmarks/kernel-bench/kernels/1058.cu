#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__device__ float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    return transpose ? matrix[col * ld + row] : matrix[row * ld + col];
}

__device__ void load_tiles(const float* __restrict__ A, const float* __restrict__ B,
                           float As[BLOCK_SIZE][BLOCK_SIZE], float Bs[BLOCK_SIZE][BLOCK_SIZE],
                           int row, int col, int t, int M, int N, int K, int lda, int ldb,
                           bool transA, bool transB) {
    const int k_a = t * BLOCK_SIZE + threadIdx.x;
    const int k_b = t * BLOCK_SIZE + threadIdx.y;
    
    const bool valid_a = (row < M) && (k_a < K);
    const bool valid_b = (col < N) && (k_b < K);
    
    As[threadIdx.y][threadIdx.x] = valid_a ? get_element(A, row, k_a, lda, transA) : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = valid_b ? get_element(B, k_b, col, ldb, transB) : 0.0f;
}

__device__ __forceinline__ float compute_partial_product(float As[BLOCK_SIZE][BLOCK_SIZE], float Bs[BLOCK_SIZE][BLOCK_SIZE]) {
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += 4) {
        sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        sum += As[threadIdx.y][k+1] * Bs[k+1][threadIdx.x];
        sum += As[threadIdx.y][k+2] * Bs[k+2][threadIdx.x];
        sum += As[threadIdx.y][k+3] * Bs[k+3][threadIdx.x];
    }
    return sum;
}

__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K,
                              int lda, int ldb, int ldc,
                              bool transA, bool transB) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    float acc = 0.0f;

    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        load_tiles(A, B, As, Bs, row, col, t, M, N, K, lda, ldb, transA, transB);
        __syncthreads();
        
        acc += compute_partial_product(As, Bs);
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * ldc + col] = acc;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    if (!A.is_cuda() || !B.is_cuda())
        throw std::invalid_argument("Inputs must be CUDA tensors");
    if (A.dim() != 2 || B.dim() != 2)
        throw std::invalid_argument("Inputs must be 2D");

    int64_t M, N, K;
    bool transA = false, transB = false;
    int lda, ldb, ldc;

    const auto A_rows = A.size(0), A_cols = A.size(1);
    const auto B_rows = B.size(0), B_cols = B.size(1);

    if (A_rows >= A_cols && B_rows == A_cols) {
        M = A_rows; K = A_cols; N = B_cols;
        lda = A.stride(0); ldb = B.stride(0);
    } else if (A_cols > A_rows && B_rows == A_rows) {
        transA = true; M = A_cols; K = A_rows; N = B_cols;
        lda = A.stride(1); ldb = B.stride(0);
    } else if (A_rows >= A_cols && B_cols == A_cols) {
        transB = true; M = A_rows; K = A_cols; N = B_rows;
        lda = A.stride(0); ldb = B.stride(1);
    } else if (A_cols > A_rows && B_cols == A_rows) {
        transA = transB = true; M = A_cols; K = A_rows; N = B_rows;
        lda = A.stride(1); ldb = B.stride(1);
    } else {
        throw std::invalid_argument("Dimensions mismatch");
    }

    auto C = torch::empty({M, N}, A.options());
    ldc = N;

    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
                                  M, N, K, lda, ldb, ldc, transA, transB);
    
    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Optimized tall-skinny matmul with predicated loading");
}