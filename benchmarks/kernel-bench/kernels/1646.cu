#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_SIZE 32

__global__ void upper_triangular_matmul_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             const int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by + ty;
    int col = bx + tx;
    
    float sum = 0.0f;
    
    // Only compute if we're in the upper triangular region
    if (row < N && col < N && row <= col) {
        // Number of tiles needed
        int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
        
        for (int tile = 0; tile < numTiles; tile++) {
            int tileStart = tile * TILE_SIZE;
            
            // Collaborative loading of A and B tiles into shared memory
            if (row < N && (tileStart + tx) < N) {
                As[ty][tx] = A[row * N + (tileStart + tx)];
            } else {
                As[ty][tx] = 0.0f;
            }
            
            if ((tileStart + ty) < N && col < N) {
                Bs[ty][tx] = B[(tileStart + ty) * N + col];
            } else {
                Bs[ty][tx] = 0.0f;
            }
            
            __syncthreads();
            
            // Compute partial sum for this tile
            // Only consider elements that contribute to upper triangular result
            int kStart = max(tileStart, row);
            int kEnd = min(tileStart + TILE_SIZE, col);
            
            for (int k = 0; k <= (kEnd - kStart); k++) {
                if ((kStart + k) >= row && (kStart + k) <= col) {
                    sum += As[ty][k] * Bs[k][tx];
                }
            }
            
            __syncthreads();
        }
        
        // Write result
        C[row * N + col] = sum;
    }
}

torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    upper_triangular_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Optimized upper triangular matrix multiplication");
}