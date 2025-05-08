#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define WORK_PER_THREAD 4

__device__ __forceinline__ float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    return transpose ? matrix[col * ld + row] : matrix[row * ld + col];
}

__global__ void strided_matmul_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K,
                                    int lda, int ldb, int ldc,
                                    bool transA, bool transB) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * blockDim.y;
    
    // Calculate base indices for this block
    const int block_row = blockIdx.y * BLOCK_SIZE;
    const int block_col = blockIdx.x * BLOCK_SIZE;

    // Each thread processes multiple elements using stride loops
    float thread_results[WORK_PER_THREAD];
    #pragma unroll
    for (int w = 0; w < WORK_PER_THREAD; w++) {
        thread_results[w] = 0.0f;
    }

    // Calculate row and col indices for each work item
    int rows[WORK_PER_THREAD], cols[WORK_PER_THREAD];
    #pragma unroll
    for (int w = 0; w < WORK_PER_THREAD; w++) {
        const int work_id = tid + w * total_threads;
        rows[w] = block_row + (work_id / BLOCK_SIZE);
        cols[w] = block_col + (work_id % BLOCK_SIZE);
    }

    // Loop over tiles
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Collaborative loading of tiles into shared memory
        for (int i = tid; i < BLOCK_SIZE * BLOCK_SIZE; i += total_threads) {
            const int local_row = i / BLOCK_SIZE;
            const int local_col = i % BLOCK_SIZE;
            
            const int global_row = block_row + local_row;
            const int global_col = t * BLOCK_SIZE + local_col;

            if (global_row < M && global_col < K)
                As[local_row][local_col] = get_element(A, global_row, global_col, lda, transA);
            else
                As[local_row][local_col] = 0.0f;

            const int b_row = t * BLOCK_SIZE + local_row;
            const int b_col = block_col + local_col;

            if (b_row < K && b_col < N)
                Bs[local_row][local_col] = get_element(B, b_row, b_col, ldb, transB);
            else
                Bs[local_row][local_col] = 0.0f;
        }
        
        __syncthreads();

        // Compute partial products for all work items
        #pragma unroll
        for (int w = 0; w < WORK_PER_THREAD; w++) {
            if (rows[w] < M && cols[w] < N) {
                const int local_row = rows[w] - block_row;
                const int local_col = cols[w] - block_col;
                
                float sum = 0.0f;
                #pragma unroll
                for (int k = 0; k < BLOCK_SIZE; ++k) {
                    sum += As[local_row][k] * Bs[k][local_col];
                }
                thread_results[w] += sum;
            }
        }

        __syncthreads();
    }

    // Write results back to global memory
    #pragma unroll
    for (int w = 0; w < WORK_PER_THREAD; w++) {
        if (rows[w] < M && cols[w] < N) {
            C[rows[w] * ldc + cols[w]] = thread_results[w];
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

    strided_matmul_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb, ldc,
        transA, transB);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication with strided loop optimization (CUDA)");
}