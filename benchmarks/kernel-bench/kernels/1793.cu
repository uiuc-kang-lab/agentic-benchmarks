#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile size for shared memory
#define TILE_SIZE 16
#define UNROLL_FACTOR 4

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    float sum = 0.0f;
    
    // Only compute for lower triangular part
    if (row < N && col < N && row >= col) {
        // Loop over tiles
        for (int t = col / TILE_SIZE; t <= row / TILE_SIZE; t++) {
            // Load tile into shared memory
            if (row < N && (t * TILE_SIZE + tx) < N) {
                As[ty][tx] = A[row * N + (t * TILE_SIZE + tx)];
            } else {
                As[ty][tx] = 0.0f;
            }
            
            if ((t * TILE_SIZE + ty) < N && col < N) {
                Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
            } else {
                Bs[ty][tx] = 0.0f;
            }
            
            __syncthreads();
            
            // Compute partial dot product for this tile
            if (row < N && col < N && row >= col) {
                int k_start = max(t * TILE_SIZE, col);
                int k_end = min((t + 1) * TILE_SIZE - 1, row);
                
                // Main loop with unrolling
                int k = k_start;
                #pragma unroll UNROLL_FACTOR
                for (; k <= k_end - (UNROLL_FACTOR-1); k += UNROLL_FACTOR) {
                    sum += As[ty][k % TILE_SIZE] * Bs[k % TILE_SIZE][tx];
                    sum += As[ty][(k+1) % TILE_SIZE] * Bs[(k+1) % TILE_SIZE][tx];
                    sum += As[ty][(k+2) % TILE_SIZE] * Bs[(k+2) % TILE_SIZE][tx];
                    sum += As[ty][(k+3) % TILE_SIZE] * Bs[(k+3) % TILE_SIZE][tx];
                }
                
                // Handle remaining elements
                for (; k <= k_end; k++) {
                    sum += As[ty][k % TILE_SIZE] * Bs[k % TILE_SIZE][tx];
                }
            }
            
            __syncthreads();
        }
        
        if (row < N && col < N) {
            if (row < col) {
                C[row * N + col] = 0.0f;
            } else {
                C[row * N + col] = sum;
            }
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                   (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}