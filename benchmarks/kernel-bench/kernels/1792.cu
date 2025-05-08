#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define PADDING 1  // Add padding to avoid bank conflicts

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N) {
    // Shared memory with padding to avoid bank conflicts
    __shared__ float As[2][TILE_SIZE][TILE_SIZE + PADDING];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + PADDING];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    float sum = 0.0f;
    
    // Double buffering indices
    int current = 0;
    int next = 1;
    
    if (row < N && col < N && row >= col) {
        // Number of tiles needed
        int numTiles = (row - col + TILE_SIZE - 1) / TILE_SIZE;
        
        // Pre-load first tile
        if (row < N && (col + tx) <= row) {
            As[current][ty][tx] = A[row * N + (col + tx)];
        } else {
            As[current][ty][tx] = 0.0f;
        }
        
        if ((col + ty) < N && col < N) {
            Bs[current][ty][tx] = B[(col + ty) * N + col];
        } else {
            Bs[current][ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Main loop over tiles
        for (int t = 0; t < numTiles; t++) {
            // Pre-load next tile if available
            if (t < numTiles - 1) {
                int nextTileOffset = (t + 1) * TILE_SIZE;
                if (row < N && (col + nextTileOffset + tx) <= row) {
                    As[next][ty][tx] = A[row * N + (col + nextTileOffset + tx)];
                } else {
                    As[next][ty][tx] = 0.0f;
                }
                
                if ((col + nextTileOffset + ty) < N && col < N) {
                    Bs[next][ty][tx] = B[(col + nextTileOffset + ty) * N + col];
                } else {
                    Bs[next][ty][tx] = 0.0f;
                }
            }
            
            // Compute using current tile
            #pragma unroll 16
            for (int k = 0; k < TILE_SIZE; k++) {
                if ((t * TILE_SIZE + k) <= (row - col)) {
                    sum += As[current][ty][k] * Bs[current][k][tx];
                }
            }
            
            __syncthreads();
            
            // Swap buffers
            current = 1 - current;
            next = 1 - next;
        }
        
        // Write result
        if (row >= col) {
            C[row * N + col] = sum;
        }
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

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Shared memory optimized triangular matrix multiplication (CUDA)");
}