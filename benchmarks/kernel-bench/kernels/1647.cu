#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void upper_triangular_matmul_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             const int N) {
    const int TILE_SIZE = 16;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Block row and column
    const int by = blockIdx.y;
    const int bx = blockIdx.x;
    
    // Thread row and column within block
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    
    // Global row and column indices
    const int row = by * TILE_SIZE + ty;
    const int col = bx * TILE_SIZE + tx;

    // Early exit if thread would work on lower triangular part
    if (row > col || row >= N || col >= N) return;

    float sum = 0.0f;
    
    // Number of tiles needed
    const int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    // Iterate over tiles
    for (int t = 0; t < numTiles; ++t) {
        // Load tile data into shared memory
        const int tileStart = t * TILE_SIZE;
        
        if ((row < N) && (tileStart + tx < N) && (tileStart + tx >= row)) {
            As[ty][tx] = A[row * N + (tileStart + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if ((tileStart + ty < N) && (col < N) && (tileStart + ty <= col)) {
            Bs[ty][tx] = B[(tileStart + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product for this tile
        if (row < N && col < N && row <= col) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                if (tileStart + k >= row && tileStart + k <= col) {
                    sum += As[ty][k] * Bs[k][tx];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < N && col < N && row <= col) {
        C[row * N + col] = sum;
    }
}

torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);
    
    dim3 threads(32, 32);
    dim3 blocks((N + threads.x - 1) / threads.x, 
                (N + threads.y - 1) / threads.y);
    
    upper_triangular_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        N
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Upper triangular matrix multiplication");
}