#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel with optimized synchronization for lower triangular matrix multiplication
__global__ void sync_optimized_triangular_mm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int N) {
    
    // Block size for shared memory tile
    const int BLOCK_SIZE = 16;
    
    // Shared memory for caching B matrix block
    __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Accumulator for dot product
    float sum = 0.0f;
    
    // Only compute if we're in the lower triangular part
    if (row < N && col < N && row >= col) {
        // Process matrix multiplication in tiles
        for (int tile = col; tile <= row; tile += BLOCK_SIZE) {
            // Load B matrix block into shared memory
            if (tile + threadIdx.x < N && row >= tile) {
                B_shared[threadIdx.y][threadIdx.x] = B[(tile + threadIdx.y) * N + col];
            } else {
                B_shared[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            // Single sync point per tile - only when shared memory is loaded
            __syncthreads();
            
            // Compute partial dot product for this tile
            #pragma unroll
            for (int k = 0; k < BLOCK_SIZE && tile + k <= row; k++) {
                sum += A[row * N + (tile + k)] * B_shared[k][threadIdx.x];
            }
            
            // No sync needed here since we're done with shared memory for this tile
        }
        
        // Write result to global memory
        C[row * N + col] = sum;
    } else if (row < N && col < N) {
        // Upper triangular part is set to zero
        C[row * N + col] = 0.0f;
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

    const int BLOCK_SIZE = 16;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    sync_optimized_triangular_mm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Sync-optimized triangular matrix multiplication (CUDA)");
}