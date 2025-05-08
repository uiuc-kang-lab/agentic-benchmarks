#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cassert>

// Define maximum allowed matrix dimension so that matrix B fits in constant memory.
// For float, constant memory size is typically 64KB. Here, MAX_N*MAX_N*sizeof(float) must be <= 64KB.
// With MAX_N = 128, we have 128*128*4 = 65536 bytes.
#define MAX_N 128

// Declare constant memory for storing matrix B (read-only).
__constant__ float const_B[MAX_N * MAX_N];

// Kernel computes the upper triangular matrix multiplication (C = A * B) for elements where row <= col.
// Matrix A is read from global memory (using __ldg for read-only cache), and matrix B is loaded from constant memory.
// For each element C[row, col], we compute the sum for k from row to col (the upper triangular region).
__global__ void constant_mem_upper_triangular_kernel(const float* __restrict__ A, float* __restrict__ C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N && row <= col) {
        float sum = 0.0f;
        // Compute the dot product over the range [row, col] complying with the upper triangular property.
        for (int k = row; k <= col; ++k) {
            float a_val = __ldg(&A[row * N + k]);
            float b_val = __ldg(&const_B[k * N + col]);
            sum += a_val * b_val;
        }
        C[row * N + col] = sum;
    }
}

// Host function that sets up the kernel call. It copies matrix B into constant memory
// and then launches the kernel which computes the multiplication for the upper triangular part.
// Note: This implementation assumes that the matrix dimension N does not exceed MAX_N so that matrix B fits in constant memory.
torch::Tensor constant_mem_upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    // Ensure N does not exceed the maximum allowed dimension for constant memory usage.
    assert(N <= MAX_N && "Matrix dimension exceeds constant memory capacity");

    // Allocate output tensor C and initialize with zeros.
    auto C = torch::zeros_like(A);

    // Copy the entire matrix B to constant memory
    cudaMemcpyToSymbol(const_B, B.data_ptr<float>(), N * N * sizeof(float));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    constant_mem_upper_triangular_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &constant_mem_upper_triangular_matmul, "Constant memory optimized upper triangular matrix multiplication");
}
