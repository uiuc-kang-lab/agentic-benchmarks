#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile size for shared memory optimization
#define TILE_SIZE 16

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Only compute if we're in the lower triangular part
    if (row >= col && row < N && col < N) {
        // Loop over tiles
        const int numTiles = (row - col + TILE_SIZE - 1) / TILE_SIZE;
        
        for (int t = 0; t < numTiles; t++) {
            const int tileStart = col + t * TILE_SIZE;
            
            // Load tile into shared memory
            if (row < N && (tileStart + threadIdx.x) <= row) {
                As[threadIdx.y][threadIdx.x] = A[row * N + tileStart + threadIdx.x];
            } else {
                As[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            if ((tileStart + threadIdx.y) <= row && col < N) {
                Bs[threadIdx.y][threadIdx.x] = B[(tileStart + threadIdx.y) * N + col];
            } else {
                Bs[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            __syncthreads();
            
            // Compute partial sum for this tile
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; k++) {
                if (tileStart + k <= row) {
                    sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                }
            }
            
            __syncthreads();
        }
        
        C[row * N + col] = sum;
    } else if (row < col && row < N && col < N) {
        // Upper triangular part is zero
        C[row * N + col] = 0.0f;
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

    const int N = A.size(0);
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