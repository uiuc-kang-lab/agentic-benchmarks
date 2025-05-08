#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define UNROLL_FACTOR 4
#define BLOCK_SIZE 16

__global__ void triangular_mm_kernel_unrolled(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              const int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col <= row) {
        float sum = 0.0f;
        
        // Calculate starting point for aligned access
        int k_start = col;
        int k_end = row;
        
        // Calculate number of complete unrolled iterations
        int k_main = k_start + ((k_end - k_start) / UNROLL_FACTOR) * UNROLL_FACTOR;
        
        // Main loop with manual unrolling
        #pragma unroll
        for (int k = k_start; k < k_main; k += UNROLL_FACTOR) {
            sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
            sum += __ldg(&A[row * N + (k+1)]) * __ldg(&B[(k+1) * N + col]);
            sum += __ldg(&A[row * N + (k+2)]) * __ldg(&B[(k+2) * N + col]);
            sum += __ldg(&A[row * N + (k+3)]) * __ldg(&B[(k+3) * N + col]);
        }
        
        // Handle remaining elements
        #pragma unroll
        for (int k = k_main; k <= k_end; k++) {
            sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
        }
        
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

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                   (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel with optimized configuration
    triangular_mm_kernel_unrolled<<<numBlocks, threadsPerBlock>>>(
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