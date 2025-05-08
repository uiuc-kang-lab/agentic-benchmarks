#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define ELEMENTS_PER_THREAD 4

__device__ float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    return transpose ? matrix[col * ld + row] : matrix[row * ld + col];
}

__global__ void matmul_kernel_balanced(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int M, int N, int K,
                                      int lda, int ldb, int ldc,
                                      bool transA, bool transB) {
    // Each thread computes multiple elements
    int block_row = blockIdx.y * (BLOCK_SIZE * ELEMENTS_PER_THREAD);
    int block_col = blockIdx.x * BLOCK_SIZE;
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    __shared__ float As[ELEMENTS_PER_THREAD][BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    float C_values[ELEMENTS_PER_THREAD] = {0.0f};

    // Process multiple rows per thread
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Load B tile into shared memory
        if (t * BLOCK_SIZE + thread_row < K && block_col + thread_col < N) {
            Bs[thread_row][thread_col] = get_element(B, 
                t * BLOCK_SIZE + thread_row,
                block_col + thread_col,
                ldb, transB);
        } else {
            Bs[thread_row][thread_col] = 0.0f;
        }

        // Load multiple A tiles into shared memory
        #pragma unroll
        for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
            int row = block_row + e * BLOCK_SIZE + thread_row;
            if (row < M && t * BLOCK_SIZE + thread_col < K) {
                As[e][thread_row][thread_col] = get_element(A,
                    row,
                    t * BLOCK_SIZE + thread_col,
                    lda, transA);
            } else {
                As[e][thread_row][thread_col] = 0.0f;
            }
        }

        __syncthreads();

        // Compute multiple elements per thread
        #pragma unroll
        for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
            #pragma unroll
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                C_values[e] += As[e][thread_row][k] * Bs[k][thread_col];
            }
        }

        __syncthreads();
    }

    // Store results
    #pragma unroll
    for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
        int row = block_row + e * BLOCK_SIZE + thread_row;
        int col = block_col + thread_col;
        if (row < M && col < N) {
            C[row * ldc + col] = C_values[e];
        }
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("Input tensors must be on CUDA devices");
    }

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);

    bool transA = false, transB = false;
    int lda = A.stride(0), ldb = B.stride(0), ldc = N;

    auto C = torch::empty({M, N}, A.options());

    // Adjust grid dimensions for multiple elements per thread
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (M + (BLOCK_SIZE * ELEMENTS_PER_THREAD) - 1) / (BLOCK_SIZE * ELEMENTS_PER_THREAD));

    matmul_kernel_balanced<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb, ldc,
        transA, transB);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication with balanced workload (CUDA)");
}