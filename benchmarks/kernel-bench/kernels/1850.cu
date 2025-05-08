#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define MAX_MATRIX_DIM 8192

__constant__ int d_N;
__constant__ float d_zero = 0.0f;

__global__ void triangular_mm_kernel_coalesced(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < d_N && col <= row) {
        float sum = 0.f;
        #pragma unroll 8
        for (int k = col; k <= row; k += BLOCK_SIZE) {
            for (int i = 0; i < BLOCK_SIZE && k + i <= row; ++i) {
                sum += A[row * d_N + (k + i)] * B[(k + i) * d_N + col];
            }
        }
        C[row * d_N + col] = sum;
    } else if (row < d_N && col < d_N) {
        C[row * d_N + col] = d_zero;
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
    TORCH_CHECK(A.size(0) <= MAX_MATRIX_DIM, "Matrix dimension exceeds maximum supported size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    cudaMemcpyToSymbol(d_N, &N, sizeof(int));

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                   (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaFuncSetCacheConfig(triangular_mm_kernel_coalesced, cudaFuncCachePreferL1);

    triangular_mm_kernel_coalesced<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>()
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}