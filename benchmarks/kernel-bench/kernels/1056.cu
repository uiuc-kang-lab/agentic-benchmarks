#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K,
                              int lda, int ldb, int ldc,
                              bool transA, bool transB) {
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    
    const int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;

    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        if (row < M && t * BLOCK_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = transA ? 
                A[(t * BLOCK_SIZE + threadIdx.x) * lda + row] :
                A[row * lda + t * BLOCK_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && t * BLOCK_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = transB ?
                B[col * ldb + t * BLOCK_SIZE + threadIdx.y] :
                B[(t * BLOCK_SIZE + threadIdx.y) * ldb + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        float local_sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            local_sum += As[threadIdx.y][k + laneId % (WARP_SIZE/4)] * 
                        Bs[k + laneId % (WARP_SIZE/4)][threadIdx.x];
        }

        sum += warp_reduce_sum(local_sum);

        __syncthreads();
    }

    if (row < M && col < N) {
        if (laneId == 0) {
            C[row * ldc + col] = sum;
        }
    }
}

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

    if (A_rows >= A_cols && B_rows == A_cols) {
        M = A_rows; K = A_cols; N = B_cols;
        lda = A.stride(0); ldb = B.stride(0);
    } else if (A_cols > A_rows && B_rows == A_rows) {
        transA = true;
        M = A_cols; K = A_rows; N = B_cols;
        lda = A.stride(1); ldb = B.stride(0);
    } else if (A_rows >= A_cols && B_cols == A_cols) {
        transB = true;
        M = A_rows; K = A_cols; N = B_rows;
        lda = A.stride(0); ldb = B.stride(1);
    } else if (A_cols > A_rows && B_cols == A_rows) {
        transA = true; transB = true;
        M = A_cols; K = A_rows; N = B_rows;
        lda = A.stride(1); ldb = B.stride(1);
    } else {
        throw std::invalid_argument("Incompatible matrix dimensions");
    }

    ldc = N;
    auto C = torch::empty({M, N}, A.options());

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb, ldc,
        transA, transB);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Warp-shuffle matrix multiplication (CUDA)");
}