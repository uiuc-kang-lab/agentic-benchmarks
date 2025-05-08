#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel using 2D thread blocks and improved memory access patterns
__global__ void triangular_mm_kernel_2d(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    // Use 16x16 thread blocks for better occupancy
    const int BLOCK_SIZE = 16;
    
    // Calculate global row and column indices
    const int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    // Shared memory for block-level data
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    float sum = 0.0f;
    
    // Only compute for lower triangular portion
    if (row < N && col <= row) {
        // Process the matrix in BLOCK_SIZE x BLOCK_SIZE tiles
        const int numTiles = (row - col + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        for (int t = 0; t < numTiles; ++t) {
            const int tileStart = col + t * BLOCK_SIZE;
            
            // Collaboratively load tile into shared memory
            if (tileStart + threadIdx.x <= row && row < N) {
                As[threadIdx.y][threadIdx.x] = A[row * N + tileStart + threadIdx.x];
            } else {
                As[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            if (tileStart + threadIdx.y <= row && col < N) {
                Bs[threadIdx.y][threadIdx.x] = B[(tileStart + threadIdx.y) * N + col];
            } else {
                Bs[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            __syncthreads();
            
            // Compute partial dot product for this tile
            #pragma unroll
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                if (tileStart + k <= row) {
                    sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                }
            }
            
            __syncthreads();
        }
        
        // Write result to global memory
        if (row < N && col <= row) {
            C[row * N + col] = sum;
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    // Use 16x16 thread blocks
    const int BLOCK_SIZE = 16;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    triangular_mm_kernel_2d<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized triangular matrix multiplication (CUDA)");
}