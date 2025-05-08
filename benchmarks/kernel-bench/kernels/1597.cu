#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

#define TILE_SIZE 16

__global__ void upper_triangular_matmul_kernel(const float* A, const float* B, float* C, int N, int chunk_start) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    float sum = 0.0f;
    
    // Calculate number of tiles needed
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    if (row < N && col < N && row <= col) {
        // Process one tile at a time
        for (int t = chunk_start / TILE_SIZE; t < min((chunk_start + TILE_SIZE) / TILE_SIZE, num_tiles); ++t) {
            // Collaborative loading of tiles into shared memory
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
            
            // Compute partial sum for this tile
            if (row < N && col < N && row <= col) {
                for (int k = 0; k < TILE_SIZE; ++k) {
                    if ((t * TILE_SIZE + k) >= row && (t * TILE_SIZE + k) <= col) {
                        sum += As[ty][k] * Bs[k][tx];
                    }
                }
            }
            
            __syncthreads();
        }
        
        // Write result
        if (row < N && col < N && row <= col) {
            atomicAdd(&C[row * N + col], sum);
        }
    }
}

torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);

    auto C = torch::zeros_like(A);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    upper_triangular_matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Upper triangular matrix multiplication");
}