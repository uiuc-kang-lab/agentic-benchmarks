#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define TILE_STRIDE 4

__device__ __forceinline__ float4 load_vector(const float* ptr, int idx) {
    return reinterpret_cast<const float4*>(ptr)[idx];
}

__device__ void load_tile(const float* matrix, float* shared,
                          int row_start, int col_start,
                          int M, int K, int lda, bool trans) {
    #pragma unroll
    for (int i = 0; i < TILE_STRIDE; ++i) {
        int row = row_start + threadIdx.y * TILE_STRIDE + i;
        int col = col_start + threadIdx.x;
        
        float val = 0.0f;
        if (row < M && col < K)
            val = trans ? matrix[col * lda + row] : matrix[row * lda + col];
        
        shared[(threadIdx.y * TILE_STRIDE + i) * BLOCK_SIZE + threadIdx.x] = val;
    }
}

__device__ void multiply_tiles(const float* A_tile, const float* B_tile,
                               float accumulators[TILE_STRIDE][TILE_STRIDE],
                               int tile_size) {
    #pragma unroll
    for (int k = 0; k < tile_size; ++k) {
        float a_frag[TILE_STRIDE];
        float b_frag[TILE_STRIDE];

        #pragma unroll
        for (int i = 0; i < TILE_STRIDE; ++i)
            a_frag[i] = A_tile[i * BLOCK_SIZE + k];

        #pragma unroll
        for (int j = 0; j < TILE_STRIDE; ++j)
            b_frag[j] = B_tile[k * BLOCK_SIZE + j];

        #pragma unroll
        for (int i = 0; i < TILE_STRIDE; ++i) {
            #pragma unroll
            for (int j = 0; j < TILE_STRIDE; ++j) {
                accumulators[i][j] += a_frag[i] * b_frag[j];
            }
        }
    }
}

__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K,
                              int lda, int ldb, int ldc,
                              bool transA, bool transB) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + 1];

    float accum[TILE_STRIDE][TILE_STRIDE] = {0};

    int tile_blocks = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int row_tile = (blockIdx.y * BLOCK_SIZE + threadIdx.y * TILE_STRIDE) * TILE_STRIDE;
    int col_tile = blockIdx.x * BLOCK_SIZE + threadIdx.x * TILE_STRIDE;

    for (int t = 0; t < tile_blocks; ++t) {
        load_tile(A, &As[0][0],
                 row_tile, t * BLOCK_SIZE,
                 M, K, lda, transA);

        load_tile(B, &Bs[0][0],
                 t * BLOCK_SIZE, col_tile,
                 K, N, ldb, transB);

        __syncthreads();
        multiply_tiles(&As[threadIdx.y * TILE_STRIDE][0],
                      &Bs[0][threadIdx.x * TILE_STRIDE],
                      accum, BLOCK_SIZE);
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TILE_STRIDE; ++i) {
        int row = row_tile + threadIdx.y * TILE_STRIDE + i;
        #pragma unroll
        for (int j = 0; j < TILE_STRIDE; ++j) {
            int col = col_tile + threadIdx.x * TILE_STRIDE + j;
            if (row < M && col < N)
                C[row * ldc + col] = accum[i][j];
        }
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // ... (same shape handling and kernel configuration as before)
    const dim3 block(BLOCK_SIZE / TILE_STRIDE, BLOCK_SIZE / TILE_STRIDE);
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    const dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    matmul_kernel<<<grid, block>>>(...);
    // ... (same synchronization and return as before)
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Vectorized tile processing matmul");
}