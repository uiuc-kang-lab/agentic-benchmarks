#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#define BLOCK_SIZE 32
#define TILE_SIZE 32

__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void optimized_upper_triangular_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.x * TILE_SIZE + threadIdx.y;
    int col = blockIdx.y * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Iterate over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile into shared memory
        int tileStart = t * TILE_SIZE;
        if (row < N && (tileStart + threadIdx.x) < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + (tileStart + threadIdx.x)];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if ((tileStart + threadIdx.y) < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(tileStart + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product for this tile
        if (row < N && col < N && row <= col) {
            int kStart = max(tileStart, row);
            int kEnd = min(tileStart + TILE_SIZE, col + 1);
            
            #pragma unroll
            for (int k = kStart; k < kEnd; ++k) {
                sum += As[threadIdx.y][k - tileStart] * 
                       Bs[k - tileStart][threadIdx.x];
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
    
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    optimized_upper_triangular_matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Optimized upper triangular matrix multiplication");
}