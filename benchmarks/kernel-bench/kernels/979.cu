#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 16
#define ELEMENTS_PER_THREAD 4
#define NUM_STREAMS 4

__device__ float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    return transpose ? matrix[col * ld + row] : matrix[row * ld + col];
}

__global__ void matmul_kernel_streamed(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int M, int N, int K,
                                        int lda, int ldb, int ldc,
                                        int m_offset,
                                        bool transA, bool transB) {
    int block_row = blockIdx.y * (BLOCK_SIZE * ELEMENTS_PER_THREAD) + m_offset;
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

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);

    bool transA = false, transB = false;
    int lda = A.stride(0), ldb = B.stride(0), ldc = N;

    auto C = torch::empty({M, N}, A.options());

    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int rows_per_stream = (M + NUM_STREAMS - 1) / NUM_STREAMS;
    rows_per_stream = ((rows_per_stream + BLOCK_SIZE * ELEMENTS_PER_THREAD - 1) / (BLOCK_SIZE * ELEMENTS_PER_THREAD)) * (BLOCK_SIZE * ELEMENTS_PER_THREAD);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        int m_offset = i * rows_per_stream;
        int current_M = std::min(rows_per_stream, static_cast<int>(M - m_offset));
        
        if (current_M <= 0) continue;

        dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (current_M + BLOCK_SIZE * ELEMENTS_PER_THREAD - 1) / (BLOCK_SIZE * ELEMENTS_PER_THREAD));

        matmul_kernel_streamed<<<gridDim, blockDim, 0, streams[i]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, N, K,
            lda, ldb, ldc,
            m_offset,
            transA, transB);
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication with streamed execution (CUDA)");
}
