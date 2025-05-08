#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 8
#define VECTOR_SIZE 4  // Using float4 for vectorized loads

__global__ void block_tuned_upper_triangular_kernel(const float* __restrict__ A,
                                                   const float* __restrict__ B,
                                                   float* __restrict__ C,
                                                   const int N) {
    // Calculate global indices
    const int row = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;
    const int col = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
    
    if (row < N && col < N && row <= col) {
        float sum = 0.0f;
        
        // Process elements in chunks of 4 when possible
        const int aligned_start = (row + 3) & ~3;  // Round up to next multiple of 4
        const int aligned_end = (col + 1) & ~3;    // Round down to multiple of 4
        
        // Handle initial unaligned elements
        for (int k = row; k < min(aligned_start, col + 1); k++) {
            sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
        }
        
        // Main vectorized loop
        for (int k = aligned_start; k < aligned_end; k += 4) {
            float4 a_vec = *reinterpret_cast<const float4*>(&A[row * N + k]);
            float4 b_vec = *reinterpret_cast<const float4*>(&B[k * N + col]);
            
            sum += a_vec.x * b_vec.x;
            sum += a_vec.y * b_vec.y;
            sum += a_vec.z * b_vec.z;
            sum += a_vec.w * b_vec.w;
        }
        
        // Handle remaining elements
        for (int k = aligned_end; k <= col; k++) {
            sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
        }
        
        C[row * N + col] = sum;
    }
}

torch::Tensor block_tuned_upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    const int N = A.size(0);
    auto C = torch::zeros_like(A);
    
    // Configure grid and block dimensions
    dim3 threadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 numBlocks((N + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
                   (N + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    
    // Launch kernel with tuned block size
    block_tuned_upper_triangular_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &block_tuned_upper_triangular_matmul, "Block-tuned upper triangular matrix multiplication");
}