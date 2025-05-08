#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_DIM 32
#define BLOCK_ROWS 8
#define MAX_THREADS 1024

template <typename scalar_t>
__global__ void optimized_hybrid_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    const int N, const int M, const int K, const int L) {

    __shared__ scalar_t tile_A[TILE_DIM][TILE_DIM];
    __shared__ scalar_t tile_B[TILE_DIM][TILE_DIM];
    
    const int tile_row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
    const int tile_col = blockIdx.x * TILE_DIM + threadIdx.x;
    const int batch_idx = tile_row / M;
    const int row_idx = tile_row % M;

    scalar_t thread_results[BLOCK_ROWS] = {0};
    
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        const int tile_k = t * TILE_DIM;
        
        if (threadIdx.y < BLOCK_ROWS) {
            #pragma unroll
            for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
                int k = tile_k + threadIdx.x;
                int row = tile_row + i;
                if (k < K && row < N * M) {
                    int batch = row / M;
                    int m = row % M;
                    tile_A[threadIdx.y + i][threadIdx.x] = __ldg(&A[batch * M * K + m * K + k]);
                } else {
                    tile_A[threadIdx.y + i][threadIdx.x] = 0;
                }
            }
        }
        
        if (threadIdx.y < BLOCK_ROWS) {
            int k = tile_k + threadIdx.y;
            int col = tile_col;
            if (k < K && col < L) {
                tile_B[threadIdx.y][threadIdx.x] = __ldg(&B[k * L + col]);
            } else {
                tile_B[threadIdx.y][threadIdx.x] = 0;
            }
        }
        
        __syncthreads();
        
        if (tile_row < N * M && tile_col < L) {
            #pragma unroll
            for (int k = 0; k < TILE_DIM; ++k) {
                scalar_t bval = tile_B[k][threadIdx.x];
                #pragma unroll
                for (int i = 0; i < BLOCK_ROWS; ++i) {
                    thread_results[i] += tile_A[threadIdx.y + i][k] * bval;
                }
            }
        }
        
        __syncthreads();
    }
    
    if (tile_col < L) {
        #pragma unroll
        for (int i = 0; i < BLOCK_ROWS; ++i) {
            int row = tile_row + i;
            if (row < N * M) {
                output[row * L + tile_col] = thread_results[i];
            }
        }
    }
}