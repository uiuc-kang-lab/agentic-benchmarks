#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel with shared memory and block-level optimization
template <typename scalar_t>
__global__ void module_fn_cuda_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int N, int M, int K, int L) {
    
    // Shared memory for tile-based computation
    __shared__ scalar_t A_shared[32][32];
    __shared__ scalar_t B_shared[32][32];
    
    int n = blockIdx.z;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize accumulator
    scalar_t sum = 0;
    
    // Loop over K dimension in tiles
    for (int tile = 0; tile < (K + 31) / 32; ++tile) {
        // Load tile into shared memory
        if (m < M && (tile * 32 + threadIdx.x) < K) {
            A_shared[threadIdx.y][threadIdx.x] = A[n * M * K + m * K + tile * 32 + threadIdx.x];
        } else {
            A_shared[threadIdx.y][threadIdx.x] = 0;
        }
        
        if ((tile * 32 + threadIdx.y) < K && l < L) {
            B_shared[threadIdx.y][threadIdx.x] = B[(tile * 32 + threadIdx.y) * L + l];
        } else {
            B_shared[threadIdx.y][threadIdx.x] = 0;
        }
        
        __syncthreads();
        
        // Compute partial dot product for this tile
        if (m < M && l < L) {
            #pragma unroll
            for (int k = 0; k < 32; k += 4) {
                sum += A_shared[threadIdx.y][k] * B_shared[k][threadIdx.x];
                sum += A_shared[threadIdx.y][k+1] * B_shared[k+1][threadIdx.x];
                sum += A_shared[threadIdx.y][k+2] * B_shared[k+2][threadIdx.x];
                sum += A_shared[threadIdx.y][k+3] * B_shared[k+3][threadIdx.x];
            }
        }
        
        __syncthreads();
    }
    
    // Write result
    if (m < M && l < L) {
        output[n * M * L + m * L + l] = sum;
    }
}