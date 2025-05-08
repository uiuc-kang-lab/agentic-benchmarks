#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define WARP_SIZE 32
#define STRIDE_FACTOR 4

__device__ __forceinline__ float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    return transpose ? matrix[col * ld + row] : matrix[row * ld + col];
}

__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K,
                              int lda, int ldb, int ldc,
                              bool transA, bool transB) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE * STRIDE_FACTOR];
    __shared__ float Bs[BLOCK_SIZE * STRIDE_FACTOR][BLOCK_SIZE];
    
    float acc = 0.0f;

    // Calculate stride for this thread
    const int stride = WARP_SIZE;
    const int k_start = tid % WARP_SIZE;
    
    // Process K dimension in tiles
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Load tile into shared memory
        if (row < M && (t * BLOCK_SIZE + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = get_element(A, row, t * BLOCK_SIZE + threadIdx.x, lda, transA);
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if ((t * BLOCK_SIZE + threadIdx.y) < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = get_element(B, t * BLOCK_SIZE + threadIdx.y, col, ldb, transB);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();

        // Compute partial products
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }

    // Write result
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
    const dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb, ldc,
        transA, transB);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Strided matrix multiplication (CUDA)");
}