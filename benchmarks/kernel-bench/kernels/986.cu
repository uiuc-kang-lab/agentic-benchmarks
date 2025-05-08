#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define ELEMENTS_PER_THREAD 4

__constant__ int const_dims[6];  // M, N, K, lda, ldb, ldc
__constant__ bool const_trans[2]; // transA, transB

__device__ float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    return transpose ? matrix[col * ld + row] : matrix[row * ld + col];
}

__global__ void matmul_kernel_unrolled(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C) {
    const int M = const_dims[0];
    const int N = const_dims[1];
    const int K = const_dims[2];
    const int lda = const_dims[3];
    const int ldb = const_dims[4];
    const int ldc = const_dims[5];
    const bool transA = const_trans[0];
    const bool transB = const_trans[1];

    int block_row = blockIdx.y * (BLOCK_SIZE * ELEMENTS_PER_THREAD);
    int block_col = blockIdx.x * BLOCK_SIZE;
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    __shared__ float As[ELEMENTS_PER_THREAD][BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    float C_values[ELEMENTS_PER_THREAD] = {0.0f};

    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        if (t * BLOCK_SIZE + thread_row < K && block_col + thread_col < N) {
            Bs[thread_row][thread_col] = get_element(B, 
                t * BLOCK_SIZE + thread_row,
                block_col + thread_col,
                ldb, transB);
        } else {
            Bs[thread_row][thread_col] = 0.0f;
        }

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

        #pragma unroll
        for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
            #pragma unroll
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                C_values[e] += As[e][thread_row][k] * Bs[k][thread_col];
            }
        }

        __syncthreads();
    }

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

    int dims[6];
    dims[0] = A.size(0);  // M
    dims[1] = B.size(1);  // N
    dims[2] = A.size(1);  // K
    dims[3] = A.stride(0); // lda
    dims[4] = B.stride(0); // ldb
    dims[5] = B.size(1);   // ldc

    bool trans[2] = {false, false};

    // Copy configuration to constant memory
    cudaMemcpyToSymbol(const_dims, dims, sizeof(dims));
    cudaMemcpyToSymbol(const_trans, trans, sizeof(trans));

    auto C = torch::empty({dims[0], dims[1]}, A.options());

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((dims[1] + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (dims[0] + (BLOCK_SIZE * ELEMENTS_PER_THREAD) - 1) / (BLOCK_SIZE * ELEMENTS_PER_THREAD));

    matmul_kernel_unrolled<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication with loop unrolling (CUDA)");
}