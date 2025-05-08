#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Shared memory tile size
#define TILE_SIZE 32

__global__ void optimized_triangular_mm_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             const int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_SIZE + ty;
    const int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Calculate starting point for this tile
    const int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    const int startTile = blockIdx.x; // Start from the current column tile
    
    for (int t = startTile; t <= blockIdx.y && t < numTiles; ++t) {
        // Load tiles into shared memory
        const int tileOffset = t * TILE_SIZE;
        if (row < N && (tileOffset + tx) < N) {
            As[ty][tx] = A[row * N + (tileOffset + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if ((tileOffset + ty) < N && col < N) {
            Bs[ty][tx] = B[(tileOffset + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll 8
        for (int k = 0; k < TILE_SIZE; ++k) {
            if ((tileOffset + k) <= row) {
                sum += As[ty][k] * Bs[k][tx];
            }
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < N && col < N) {
        // Only write if in lower triangle
        if (row >= col) {
            C[row * N + col] = sum;
        } else {
            C[row * N + col] = 0.0f;
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

    optimized_triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}