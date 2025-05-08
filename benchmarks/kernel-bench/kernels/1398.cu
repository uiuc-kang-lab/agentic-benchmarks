#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_DIAG_SIZE 16384  // Maximum size for diagonal matrix

// Constant memory for diagonal matrix A
__constant__ float d_A[MAX_DIAG_SIZE];

// Device function that computes the product of a diagonal element and a given value from B
__device__ inline float compute_diag_multiply(int row, float b_val) {
    return d_A[row] * b_val;
}

// CUDA kernel: each thread computes one element of the output C
// C[i,j] = A[i] * B[i,j]
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
        // Use the modular device function for the multiplication
        C[idx] = compute_diag_multiply(row, B[idx]);
    }
}

// Forward function that wraps our CUDA kernel
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch: A.size(0) must match B.size(0)");
    TORCH_CHECK(A.size(0) <= MAX_DIAG_SIZE, "Diagonal matrix size exceeds maximum supported size");

    // Ensure inputs are contiguous in memory
    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);

    // Copy the diagonal matrix A to constant memory
    cudaMemcpyToSymbol(d_A, A.data_ptr<float>(), N * sizeof(float));

    // Create an output tensor with the same options as B
    auto C = torch::empty({N, M}, B.options());

    // Configure and launch the kernel
    const int64_t threads = 256;
    const int64_t blocks = (N * M + threads - 1) / threads;
    diag_matmul_kernel<<<blocks, threads>>>(