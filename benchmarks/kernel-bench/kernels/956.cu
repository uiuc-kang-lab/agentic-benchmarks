#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define ELEMENTS_PER_THREAD 4
#define SHARED_PAD 1  // Padding to avoid bank conflicts

__device__ __forceinline__ float4 load_float4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__device__ __forceinline__ void store_float4(float* ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K,
                              int lda, int ldb, int ldc,
                              bool transA, bool transB) {
    // Padded shared memory to avoid bank conflicts
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE + SHARED_PAD];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SHARED_PAD];
    
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int row = block_row * BLOCK_SIZE + threadIdx.y;
    const int col = block_col * BLOCK_SIZE + threadIdx.x;

    // Register array to accumulate results
    float accum[ELEMENTS_PER_THREAD] = {0.0f};
    
    // Process K dimension in tiles
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Collaborative loading of A and B tiles into shared memory
        const int tile_idx = t * BLOCK_SIZE;
        
        // Each thread loads multiple elements
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE / (blockDim.x * blockDim.y); i++) {
            int linear_idx = tid + i * (blockDim.x * blockDim.y);
            int local_row = linear_idx / BLOCK_SIZE;
            int local_col = linear_idx % BLOCK_SIZE;
            
            if (row < M && tile_idx + local_col < K) {
                As[local_row][local_col] = transA ? 
                    A[(tile_idx + local_col) * lda + block_row * BLOCK_SIZE + local_row] :
                    A[(block_row * BLOCK_SIZE + local_row) * lda + tile_idx + local_col];
            } else {
                As[local_row][local_col] = 0.0f;
            }
            
            if (col < N && tile_idx + local_row < K) {
                Bs[local_row][local_col] = transB ?
                    B[(block_col * BLOCK_SIZE + local_col) * ldb + tile_idx + local_row] :
                    B[(tile_idx + local_row) * ldb + block_col * BLOCK_SIZE + local_col];
            } else {
                Bs[local_row][local_col] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute partial products
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            float a_val = As[threadIdx.y][k];
            float b_val = Bs[k][threadIdx.x];
            #pragma unroll
            for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
                accum[e] += a_val * b_val;
            }
        }
        
        __syncthreads();
    }
    
    // Write results to global memory
    if (row < M && col < N) {
        #pragma unroll
        for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
            int out_row = row + (e / ELEMENTS_PER_THREAD) * BLOCK_SIZE;
            int out_col = col + (e % ELEMENTS_PER_THREAD);
            if (out_row < M && out_col < N) {
                C[out_row * ldc + out_col] = accum[e];
            }
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
        throw std::invalid_argument("Incompatible matrix dimensions");
    }

    ldc = N;

    auto C = torch::empty({M, N}, A.options());

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE / ELEMENTS_PER_THREAD);
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
    m.def("forward", &matmul_cuda, "Matrix multiplication with shared memory optimization (CUDA)");
}