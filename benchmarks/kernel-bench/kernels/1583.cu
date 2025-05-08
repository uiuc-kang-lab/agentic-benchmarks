#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// Shared memory tile size
#define TILE_SIZE 32

__global__ void optimized_upper_triangular_kernel(const float* __restrict__ A,
                                                const float* __restrict__ B,
                                                float* __restrict__ C,
                                                const int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    // Only compute if we're in the upper triangle
    if (row < N && col < N && row <= col) {
        // Iterate over tiles
        for (int t = row / TILE_SIZE * TILE_SIZE; t <= col; t += TILE_SIZE) {
            // Load tile into shared memory
            if (threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE) {
                int shared_row = row;
                int shared_col = t + threadIdx.x;
                if (shared_row < N && shared_col < N && shared_row <= shared_col) {
                    As[threadIdx.y][threadIdx.x] = A[shared_row * N + shared_col];
                } else {
                    As[threadIdx.y][threadIdx.x] = 0.0f;
                }
                
                shared_row = t + threadIdx.y;
                shared_col = col;
                if (shared_row < N && shared_col < N) {
                    Bs[threadIdx.y][threadIdx.x] = B[shared_row * N + shared_col];
                } else {
                    Bs[threadIdx.y][threadIdx.x] = 0.0f;
                }
            }
            
            __syncthreads();
            
            // Compute partial sum for this tile
            for (int k = 0; k < TILE_SIZE; ++k) {
                int global_k = t + k;
                if (global_k >= row && global_k <= col && global_k < N) {
                    sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                }
            }
            
            __syncthreads();
        }
        
        C[row * N + col] = sum;
    }
}

torch::Tensor optimized_upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);
    
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    optimized_upper_triangular_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_upper_triangular_matmul, "Optimized upper triangular matrix multiplication");
}