#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;
    
    if (row >= N || col >= N) return;

    // Only compute elements in the lower triangle
    if (col <= row) {
        float sum = 0.0f;
        // For each element in the lower triangle, we need to compute
        // the dot product of a row of A with a column of B
        for (int k = 0; k <= row; k++) {
            // Only multiply if we're in the lower triangle of both matrices
            if (k >= col) {
                sum += A[row * N + k] * B[k * N + col];
            }
        }
        C[row * N + col] = sum;
    } else {
        // Set upper triangle to zero
        C[row * N + col] = 0.0f;
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

    // Use one block per row and as many threads as columns
    const int threadsPerBlock = 32;
    const int numBlocks = N;

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
    m.def("forward", &forward, "Shared Memory Optimized Triangular Matrix Multiplication (CUDA)");
}