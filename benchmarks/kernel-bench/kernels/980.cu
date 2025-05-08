#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define TILE_SIZE 32
#define ELEMENTS_PER_THREAD 4

// Modular device functions for better optimization and reuse
__device__ __forceinline__ float get_element(const float* __restrict__ matrix, 
                                            int row, int col, 
                                            int ld, bool transpose) {
    return transpose ? matrix[col * ld + row] : matrix[row * ld + col];
}

__device__ __forceinline__ void load_tile_A(float (&As)[TILE_SIZE][TILE_SIZE], 
                                           const float* __restrict__ A,
                                           int row, int col,
                                           int M, int K, int lda,
                                           bool transA) {
    if (row < M && col < K) {
        As[threadIdx.y][threadIdx.x] = get_element(A, row, col, lda, transA);
    } else {
        As[threadIdx.y][threadIdx.x] = 0.0f;
    }
}

__device__ __forceinline__ void load_tile_B(float (&Bs)[TILE_SIZE][TILE_SIZE],
                                           const float* __restrict__ B,
                                           int row, int col,
                                           int K, int N, int ldb,
                                           bool transB) {
    if (row < K && col < N) {
        Bs[threadIdx.y][threadIdx.x] = get_element(B, row, col, ldb, transB);
    } else {
        Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }
}

__device__ __forceinline__ void compute_tile(float (&As)[TILE_SIZE][TILE_SIZE],
                                            float (&Bs)[TILE_SIZE][TILE_SIZE],
                                            float (&accum)[ELEMENTS_PER_THREAD]) {
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        float a_val = As[threadIdx.y][k];
        #pragma unroll
        for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
            accum[e] += a_val * Bs[k][threadIdx.x + e * BLOCK_SIZE];
        }
    }
}

__device__ __forceinline__ void store_results(float* __restrict__ C,
                                             const float (&accum)[ELEMENTS_PER_THREAD],
                                             int row, int base_col,
                                             int M, int N, int ldc) {
    if (row < M) {
        #pragma unroll
        for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
            int col = base_col + e * BLOCK_SIZE;
            if (col < N) {
                C[row * ldc + col] = accum[e];
            }
        }
    }
}

__global__ void matmul_kernel_modular(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int N, int K,
                                     int lda, int ldb, int ldc,
                                     bool transA, bool transB) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    float accum[ELEMENTS_PER_THREAD] = {0.0f};
    
    int by = blockIdx.y * BLOCK_SIZE;
    int bx = blockIdx.x * (BLOCK_SIZE * ELEMENTS_PER_THREAD);
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by + ty;
    int base_col = bx + tx;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles collaboratively
        load_tile_A(As, A, row, t * TILE_SIZE + tx, M, K, lda, transA);
        
        #pragma unroll
        for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
            load_tile_B(Bs, B, t * TILE_SIZE + ty, base_col + e * BLOCK_SIZE, 
                       K, N, ldb, transB);
        }
        
        __syncthreads();
        
        compute_tile(As, Bs, accum);
        
        __syncthreads();
    }
    
    store_results(C, accum, row, base_col, M, N, ldc);
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

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + (BLOCK_SIZE * ELEMENTS_PER_THREAD) - 1) / (BLOCK_SIZE * ELEMENTS_PER_THREAD),
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_kernel_modular<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb, ldc,
        transA, transB);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Modular optimized matrix multiplication (CUDA)");
}