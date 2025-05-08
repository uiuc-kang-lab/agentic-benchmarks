#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 16
#define NUM_STREAMS 4

__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K,
                              int M_start, int M_chunk,
                              int lda, int ldb, int ldc,
                              bool transA, bool transB) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Adjust global indices based on chunk offset
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y + M_start;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    float sum = 0.0f;

    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Load tiles into shared memory
        if (row < M_start + M_chunk && t * BLOCK_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = transA ? 
                A[(t * BLOCK_SIZE + threadIdx.x) * lda + row] :
                A[row * lda + t * BLOCK_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (t * BLOCK_SIZE + threadIdx.y < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = transB ?
                B[col * ldb + t * BLOCK_SIZE + threadIdx.y] :
                B[(t * BLOCK_SIZE + threadIdx.y) * ldb + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M_start + M_chunk && col < N) {
        C[row * ldc + col] = sum;
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

    // Create CUDA streams
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Calculate chunk size for each stream
    int chunk_size = (M + NUM_STREAMS - 1) / NUM_STREAMS;
    
    // Launch kernels in different streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        int M_start = i * chunk_size;
        int M_chunk = std::min(chunk_size, static_cast<int>(M - M_start));
        
        if (M_chunk <= 0) break;

        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (M_chunk + BLOCK_SIZE - 1) / BLOCK_SIZE);

        matmul_kernel<<<gridDim, blockDim, 0, streams[i]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, N, K,
            M_start, M_chunk,
            lda, ldb, ldc,
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
    m.def("forward", &matmul_cuda, "Stream-pipelined matrix multiplication (CUDA)");
}