#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

#define BLOCK_SIZE 64  // Changing block size based on GPU architecture capabilities

__global__ void upper_triangular_matmul_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int bx = blockIdx.x * BLOCK_SIZE;
    int by = blockIdx.y * BLOCK_SIZE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by + ty;
    int col = bx + tx;
    
    float sum = 0.0f;
    
    // Only process elements in the upper triangle
    if (row <= col && row < N && col < N) {
        // Loop over blocks
        for (int k = 0; k < N; k += BLOCK_SIZE) {
            // Collaborative loading of tiles into shared memory
            if (k + tx < N && row < N)
                As[ty][tx] = A[row * N + (k + tx)];
            else
                As[ty][tx] = 0.0f;
                
            if (k + ty < N && col < N)
                Bs[ty][tx] = B[(k + ty) * N + col];
            else
                Bs[ty][tx] = 0.0f;
                
            __syncthreads();
            
            // Compute partial dot product
            for (int i = 0; i < BLOCK_SIZE; ++i) {
                int global_k = k + i;
                if (global_k >= row && global_k <= col)
                    sum += As[ty][i] * Bs[i][tx];
            }
            
            __syncthreads();
        }
        
        C[row * N + col] = sum;
    }
}

torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);
    
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    upper_triangular_matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Upper triangular matrix multiplication");
}
