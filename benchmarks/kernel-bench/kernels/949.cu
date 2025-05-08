#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define WARP_SIZE 32

__device__ float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    if (transpose)
        return matrix[col * ld + row];
    else
        return matrix[row * ld + col];
}

__inline__ __device__
float warp_reduce_sum(float val) {
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
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    
    int row = block_row * BLOCK_SIZE + thread_row;
    int col = block_col * BLOCK_SIZE + thread_col;
    
    float thread_sum = 0.0f;
    
    // Calculate number of tiles
    int num_tiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int t = 0; t < num_tiles; ++t) {
        // Collaborative loading of tiles into shared memory
        if (row < M && t * BLOCK_SIZE + thread_col < K) {
            As[thread_row][thread_col] = get_element(A, row, t * BLOCK_SIZE + thread_col, lda, transA);
        } else {
            As[thread_row][thread_col] = 0.0f;
        }
        
        if (col < N && t * BLOCK_SIZE + thread_row < K) {
            Bs[thread_row][thread_col] = get_element(B, t * BLOCK_SIZE + thread_row, col, ldb, transB);
        } else {
            Bs[thread_row][thread_col] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial products using register accumulation
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k += 4) {
            thread_sum += As[thread_row][k] * Bs[k][thread_col];
            thread_sum += As[thread_row][k+1] * Bs[k+1][thread_col];
            thread_sum += As[thread_row][k+2] * Bs[k+2][thread_col];
            thread_sum += As[thread_row][k+3] * Bs[k+3][thread_col];
        }
        
        __syncthreads();
    }
    
    // Perform warp-level reduction if needed
    if (threadIdx.x % WARP_SIZE == 0) {
        thread_sum = warp_reduce_sum(thread_sum);
    }
    
    // Write result to global memory
    if (row < M && col < N) {
        C[row * ldc + col] = thread_sum;
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
    m.def("forward", &matmul_cuda, "Matrix multiplication with shared memory reduction (CUDA)");
}