#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// Kernel with coalesced global memory accesses for the common (non-transposed) case.
// For transposed cases, the kernel still returns the correct result, though optimal coalescing is achieved when transA and transB are false.

__global__ void coalesced_matmul_kernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int M, int N, int K,
                                        int lda, int ldb, int ldc,
                                        bool transA, bool transB) {
    // Compute the global row and column index for C
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        int tiledCol = t * BLOCK_SIZE + threadIdx.x;  // index for A
        int tiledRow = t * BLOCK_SIZE + threadIdx.y;    // index for B

        // Load tile from A with coalesced access when not transposed
        if (!transA) {
            As[threadIdx.y][threadIdx.x] = (row < M && tiledCol < K) ? A[row * lda + tiledCol] : 0.0f;
        } else {
            // When A is transposed, logical A(i,k) = physical A(k,i)
            As[threadIdx.y][threadIdx.x] = (row < M && tiledCol < K) ? A[tiledCol * lda + row] : 0.0f;
        }

        // Load tile from B with coalesced access when not transposed
        if (!transB) {
            Bs[threadIdx.y][threadIdx.x] = (tiledRow < K && col < N) ? B[tiledRow * ldb + col] : 0.0f;
        } else {
            // When B is transposed, logical B(k,j) = physical B(j,k)
            Bs[threadIdx.y][threadIdx.x] = (tiledRow < K && col < N) ? B[col * ldb + tiledRow] : 0.0f;
        }

        __syncthreads();

        // Multiply the two tiles
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * ldc + col] = sum;
    }
}

// Host function

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

    // Determine the multiplication configuration similar to the reference implementation
    if (A_rows >= A_cols && B_rows == A_cols) {
        // A is M x K, B is K x N
        M = A_rows;
        K = A_cols;
        N = B_cols;
        lda = A.stride(0);
        ldb = B.stride(0);
    } else if (A_cols > A_rows && B_rows == A_rows) {
        // A is stored transposed
        transA = true;
        M = A_cols;
        K = A_rows;
        N = B_cols;
        lda = A.stride(1);
        ldb = B.stride(0);
    } else if (A_rows >= A_cols && B_cols == A_cols) {
        // B is stored transposed
        transB = true;
        M = A_rows;
        K = A_cols;
        N = B_rows;
        lda = A.stride(0);
        ldb = B.stride(1);
    } else if (A_cols > A_rows && B_cols == A_rows) {
        // Both A and B are stored transposed
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

    coalesced_matmul_kernel<<<gridDim, blockDim>>>(
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
    m.def("forward", &matmul_cuda, "Coalesced memory access matrix multiplication (CUDA)");
}
