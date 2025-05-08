#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_DIAG_SIZE 16384  // Maximum size for diagonal matrix

__constant__ float d_A[MAX_DIAG_SIZE];  // Constant memory for diagonal matrix

__global__ void diag_matmul_kernel(
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * M;
    if (idx < total) {
        int row = idx / M;
        int col = idx % M;
        C[idx] = d_A[row] * B[idx];
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0),
                "Dimension mismatch: A.size(0) must match B.size(0)");
    TORCH_CHECK(A.size(0) <= MAX_DIAG_SIZE,
                "Diagonal matrix size exceeds maximum supported size");

    // Ensure inputs are contiguous
    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);

    // Copy diagonal matrix to constant memory
    cudaMemcpyToSymbol(d_A, A.data_ptr<float>(), N * sizeof(float), 0, cudaMemcpyHostToDevice);

    // Create output tensor
    auto C = torch::empty({N, M}, B.options());

    // Launch kernel
    const int64_t threads = 256;
    const int64_t blocks = (N * M + threads - 1) / threads;
    diag_matmul_kernel<<<blocks, threads>>>(
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Diagonal matrix multiplication of A and B on the GPU");
}