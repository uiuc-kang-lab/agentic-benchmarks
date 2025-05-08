#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 32  // Increased block size for better occupancy
#define CHUNK_SIZE 8   // For memory coalescing

// Inline function to perform a read-only load using __ldg
__device__ __forceinline__ float load_elem(const float* __restrict__ matrix, int index) {
    return __ldg(&matrix[index]);
}

// Get element with transpose handling and read-only loads
__device__ __forceinline__ float get_element(const float* __restrict__ matrix, int row, int col, 
                                           int ld, bool transpose) {
    return load_elem(matrix, transpose ? (col * ld + row) : (row * ld + col));
}

__global__ void matmul_kernel(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int K,
                            int lda, int ldb, int ldc,
                            bool transA, bool transB) {
    // Use vectorized loads where possible
    typedef float4 vec_t;

    // Shared memory with padding to avoid bank conflicts
    __shared__ __align__(16) float As[BLOCK_SIZE][BLOCK_SIZE + 2];
    __shared__ __align__(16) float Bs[BLOCK_SIZE][BLOCK_SIZE + 2];

    // Register array for accumulation
    float thread_results[CHUNK_SIZE][CHUNK_SIZE] = {0.0f};
    
    int block_row = blockIdx.y * BLOCK_SIZE;
    int block_col = blockIdx.x * BLOCK_SIZE;
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    // Loop over K dimension tiles
    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        // Collaborative loading of A and B tiles using vectorized loads where possible
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i += CHUNK_SIZE) {
            if (block_row + thread_row < M && tile * BLOCK_SIZE + i + thread_col < K) {
                As[thread_row][i + thread_col] = get_element(A, 
                    block_row + thread_row,
                    tile * BLOCK_SIZE + i + thread_col,
                    lda, transA);
            }
            
            if (tile * BLOCK_SIZE + thread_row < K && block_col + i + thread_col < N) {
                Bs[thread_row][i + thread_col] = get_element(B,
                    tile * BLOCK_SIZE + thread_row,
                    block_col + i + thread_col,
                    ldb, transB);
            }
        }
        
        __syncthreads();

        // Compute partial products with register blocking
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            float a_val = As[thread_row][k];
            #pragma unroll
            for (int i = 0; i < CHUNK_SIZE; ++i) {
                #pragma unroll
                for (int j = 0; j < CHUNK_SIZE; ++j) {
                    thread_results[i][j] += a_val * Bs[k][thread_col * CHUNK_SIZE + j];
                }
            }
        }

        __syncthreads();
    }

    // Write results to global memory with coalesced access
    #pragma unroll
    for (int i = 0; i < CHUNK_SIZE; ++i) {
        #pragma unroll
        for (int j = 0; j < CHUNK_SIZE; ++j) {
            int global_row = block_row + thread_row * CHUNK_SIZE + i;
            int global_col = block_col + thread_col * CHUNK_SIZE + j;
            if (global_row < M && global_col < N) {
                C[global_row * ldc + global_col] = thread_results[i][j];
            }
        }
    }
}