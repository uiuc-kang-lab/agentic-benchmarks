#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_DIM 16
#define BLOCK_ROWS 16
#define PADDING 1

__global__ void shared_memory_optimized_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             const int N) {
    __shared__ float As[TILE_DIM][TILE_DIM + PADDING];
    __shared__ float Bs[TILE_DIM][TILE_DIM + PADDING];
    
    const int row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
    const int col = blockIdx.x * TILE_DIM + threadIdx.x;
    
    float sum = 0.0f;
    
    // Only proceed if we're in the upper triangle
    if (row < N && col < N && row <= col) {
        // Calculate the starting tile for this row
        const int start_tile = row / TILE_DIM;
        const int end_tile = (col + TILE_DIM - 1) / TILE_DIM;
        
        for (int t = start_tile; t <= end_tile; t++) {
            // Load tile of A into shared memory
            const int tile_row = row;
            const int tile_col = t * TILE_DIM + threadIdx.x;
            if (tile_row < N && tile_col < N && threadIdx.x < TILE_DIM) {
                if (tile_row <= tile_col) {
                    As[threadIdx.y][threadIdx.x] = A[tile_row * N + tile_col];
                } else {
                    As[threadIdx.y][threadIdx.x] = 0.0f;
                }
            }
            
            // Load tile of B into shared memory
            const int b_row = t * TILE_DIM + threadIdx.y;
            const int b_col = col;
            if (b_row < N && b_col < N && threadIdx.y < TILE_DIM) {
                Bs[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
            } else {
                Bs[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            __syncthreads();
            
            // Compute partial dot product for this tile
            #pragma unroll
            for (int k = 0; k < TILE_DIM; k++) {
                const int global_k = t * TILE_DIM + k;
                if (global_k >= row && global_k <= col && global_k < N) {
                    sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                }
            }
            
            __syncthreads();
        }
        
        // Write result
        if (row < N && col < N && row <= col) {
            C[row * N + col] = sum;
        }
    }
}

torch::Tensor shared_memory_optimized_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);
    
    dim3 threadsPerBlock(TILE_DIM, BLOCK_ROWS);
    dim3 numBlocks((N + TILE_DIM - 1) / TILE_DIM,
                   (N + BLOCK_ROWS - 1) / BLOCK_ROWS);
    
    shared_memory_optimized_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_memory_optimized_matmul, "Shared memory optimized upper triangular matrix multiplication");
}