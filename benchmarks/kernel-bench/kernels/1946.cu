#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define SMALL_MATRIX_THRESHOLD 256

// Simple kernel for small matrices
__global__ void simple_triangular_mm_kernel(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N && row >= col) {
        float sum = 0.f;
        for (int k = col; k <= row; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    } else if (row < N && col < N) {
        C[row * N + col] = 0.f;
    }
}

// Optimized kernel with shared memory for large matrices
__global__ void tiled_triangular_mm_kernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.f;

    if (row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.f;
            return;
        }

        int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
        
        #pragma unroll 4
        for (int m = 0; m < numTiles; m++) {
            int k_base = m * TILE_SIZE;
            
            // Collaborative loading with vectorized reads where possible
            if ((k_base + threadIdx.x) <= row) {
                sA[threadIdx.y][threadIdx.x] = A[row * N + k_base + threadIdx.x];
            }
            if ((k_base + threadIdx.y) < N) {
                sB[threadIdx.y][threadIdx.x] = B[(k_base + threadIdx.y) * N + col];
            }
            
            __syncthreads();

            int k_start = max(k_base, col);
            int k_end = min(k_base + TILE_SIZE, row + 1);
            int local_start = k_start - k_base;
            int local_end = k_end - k_base;

            #pragma unroll 8
            for (int k = local_start; k < local_end; ++k) {
                sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
            }
            
            if (m != numTiles - 1) {
                __syncthreads();
            }
        }
        
        C[row * N + col] = sum;
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Inputs must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Input dimensions must match");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    if (N <= SMALL_MATRIX_THRESHOLD) {
        // For small matrices, use simple kernel
        dim3 threads(16, 16);
        dim3 blocks((N + 15) / 16, (N + 15) / 16);
        simple_triangular_mm_kernel<<<blocks, threads>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    } else {
        // For large matrices, use tiled kernel
        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
        tiled_triangular_mm_kernel<<<blocks, threads>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}