#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define VECTOR_SIZE 4  // Using float4 for vectorized loads

__global__ void triangular_mm_kernel_aligned(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           const int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col <= row) {
        float sum = 0.f;
        
        // Process elements in chunks of 4 when possible
        int k = col;
        int aligned_end = ((row + 1) - ((row + 1) % VECTOR_SIZE));
        
        // Vectorized loading for aligned sections
        for (; k + VECTOR_SIZE - 1 <= aligned_end && k + VECTOR_SIZE - 1 <= row; k += VECTOR_SIZE) {
            float4 a_vec = *reinterpret_cast<const float4*>(&A[row * N + k]);
            float4 b_vec = *reinterpret_cast<const float4*>(&B[k * N + col]);
            
            sum += a_vec.x * b_vec.x;
            sum += a_vec.y * b_vec.y;
            sum += a_vec.z * b_vec.z;
            sum += a_vec.w * b_vec.w;
        }
        
        // Handle remaining elements
        for (; k <= row; ++k) {
            sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
        }
        
        // Aligned store
        C[row * N + col] = sum;
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");
    
    int N = A.size(0);
    auto C = torch::empty_like(A);
    
    // Ensure matrices are aligned properly
    TORCH_CHECK((uintptr_t)A.data_ptr() % 16 == 0, "Matrix A must be 16-byte aligned");
    TORCH_CHECK((uintptr_t)B.data_ptr() % 16 == 0, "Matrix B must be 16-byte aligned");
    TORCH_CHECK((uintptr_t)C.data_ptr() % 16 == 0, "Matrix C must be 16-byte aligned");
    
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Set L1 cache preference
    cudaFuncSetCacheConfig(triangular_mm_kernel_aligned, cudaFuncCachePreferL1);
    
    triangular_mm_kernel_aligned<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}