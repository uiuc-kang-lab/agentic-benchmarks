#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 16
#define NUM_STREAMS 4

__global__ void matmul_kernel_streamed(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int N, int K,
                                     int lda, int ldb, int ldc,
                                     int m_offset,
                                     bool transA, bool transB) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y + m_offset;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float C_value = 0.0f;
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        if (row < M && t * BLOCK_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = transA ? 
                A[(t * BLOCK_SIZE + threadIdx.x) * lda + row] :
                A[row * lda + (t * BLOCK_SIZE + threadIdx.x)];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && t * BLOCK_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = transB ?
                B[col * ldb + (t * BLOCK_SIZE + threadIdx.y)] :
                B[(t * BLOCK_SIZE + threadIdx.y) * ldb + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            C_value += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * ldc + col] = C_value;
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
    int lda = A.stride(0);
    int ldb = B.stride(0);
    int ldc = N;

    auto C = torch::empty({M, N}, A.options());

    // Create CUDA streams
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Calculate rows per stream
    int rows_per_stream = (M + NUM_STREAMS - 1) / NUM_STREAMS;
    rows_per_stream = ((rows_per_stream + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    
    // Launch kernels in different streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        int m_offset = i * rows_per_stream;
        int current_M = std::min(rows_per_stream, static_cast<int>(M - m_offset));
        
        if (current_M <= 0) continue;

        dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (current_M + BLOCK_SIZE - 1) / BLOCK_SIZE);

        matmul_kernel_streamed<<<gridDim, blockDim, 0, streams[i]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, N, K,
            lda, ldb, ldc,
            m_offset,
            transA, transB);
    }

    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication with stream-based pipelining (CUDA)");
}