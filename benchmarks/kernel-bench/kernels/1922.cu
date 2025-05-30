#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32  // Increased tile size for better occupancy
#define BLOCK_ROWS 8  // Process multiple rows per thread

__global__ void optimized_triangular_mm_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             const int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int bx = blockIdx.x * TILE_SIZE;
    const int by = blockIdx.y * TILE_SIZE;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Each thread processes multiple rows for better work efficiency
    float thread_results[BLOCK_ROWS] = {0.0f};
    
    #pragma unroll
    for (int block_row = 0; block_row < BLOCK_ROWS; block_row++) {
        const int row = by + ty + block_row * (TILE_SIZE/BLOCK_ROWS);
        const int col = bx + tx;
        
        // Early exit condition for entire row block
        if (row >= N) continue;
        
        // Process only if we're in the lower triangular part
        if (row >= col && col < N) {
            // Calculate starting tile to avoid unnecessary computations
            const int start_tile = col / TILE_SIZE;
            const int end_tile = (N + TILE_SIZE - 1) / TILE_SIZE;
            
            for (int tile = start_tile; tile < end_tile; tile++) {
                // Collaborative loading of tiles using vectorized loads where possible
                if (tile * TILE_SIZE + tx < N) {
                    As[ty][tx] = A[row * N + (tile * TILE_SIZE + tx)];
                    Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
                }
                
                __syncthreads();
                
                // Compute partial sum for this tile
                #pragma unroll 8
                for (int k = 0; k < TILE_SIZE; k++) {
                    if (tile * TILE_SIZE + k < N) {
                        thread_results[block_row] += As[ty][k] * Bs[k][tx];
                    }
                }
                
                __syncthreads();
            }
            
            // Write result
            C[row * N + col] = thread_results[block_row];
        } else if (col < N) {
            // Set upper triangular part to zero
            C[row * N + col] = 0.0f;
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Inputs must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Input dimensions must match");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE/BLOCK_ROWS);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                   (N + TILE_SIZE - 1) / TILE_SIZE);

    optimized_triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized triangular matrix multiplication (CUDA)");
}