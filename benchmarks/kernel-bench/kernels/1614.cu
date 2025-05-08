#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define BLOCK_SIZE 16
#define WARP_SIZE 32

__device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void upper_triangular_matmul_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    // Only process elements in the upper triangular part
    if (row < N && col < N && row <= col) {
        // Process the matrix in BLOCK_SIZE x BLOCK_SIZE tiles
        for (int tile = 0; tile <= col/BLOCK_SIZE; ++tile) {
            // Collaborative loading of A and B tiles into shared memory
            if (threadIdx.x < BLOCK_SIZE && threadIdx.y < BLOCK_SIZE) {
                int global_row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
                int global_col = tile * BLOCK_SIZE + threadIdx.x;
                
                if (global_row < N && global_col < N) {
                    As[threadIdx.y][threadIdx.x] = A[global_row * N + global_col];
                    Bs[threadIdx.y][threadIdx.x] = B[global_col * N + blockIdx.x * BLOCK_SIZE + threadIdx.x];
                } else {
                    As[threadIdx.y][threadIdx.x] = 0.0f;
                    Bs[threadIdx.y][threadIdx.x] = 0.0f;
                }
            }
            __syncthreads();
            
            // Compute partial dot product using shared memory
            if (row < N && col < N && row <= col) {
                int tile_start = tile * BLOCK_SIZE;
                int tile_end = min((tile + 1) * BLOCK_SIZE, N);
                
                for (int k = tile_start; k < tile_end; ++k) {
                    if (k >= row) {  // Only multiply elements from upper triangular part
                        sum += As[threadIdx.y][k - tile_start] * Bs[k - tile_start][threadIdx.x];
                    }
                }
            }
            __syncthreads();
        }
        
        // Write result directly (no need for warp reduction as each thread computes its own result)
        C[row * N + col] = sum;
    }
}

torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);
    
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    upper_triangular_matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Upper triangular matrix multiplication with shared memory and warp reduction");
}