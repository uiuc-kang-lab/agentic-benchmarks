#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = tid / N;
    const int col = tid % N;
    
    if (row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.f;
        } else {
            float sum = 0.f;
            
            // Ensure k is aligned to 4-float boundary for vectorized loads
            int k_start = (col + 3) & ~3;  // Round up to nearest multiple of 4
            
            // Handle initial unaligned elements
            for (int k = col; k < min(k_start, row + 1); ++k) {
                sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
            }
            
            // Main loop with vectorized loads
            #pragma unroll 4
            for (int k = k_start; k <= (row & ~3); k += 4) {
                float4 a_vec = *reinterpret_cast<const float4*>(&A[row * N + k]);
                float4 b_vec = *reinterpret_cast<const float4*>(&B[k * N + col]);
                
                sum += a_vec.x * b_vec.x;
                sum += a_vec.y * b_vec.y;
                sum += a_vec.z * b_vec.z;
                sum += a_vec.w * b_vec.w;
            }
            
            // Handle remaining elements
            for (int k = (row & ~3) + 1; k <= row; ++k) {
                sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
            }
            
            C[row * N + col] = sum;
        }
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

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    // Optimize block size for memory alignment
    const int threadsPerBlock = 256;
    const int numElements = N * N;
    const int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
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