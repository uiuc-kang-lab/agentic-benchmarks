#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// Device function to compute the dot product for a given (row, col) in the upper triangular matrix
// It computes the sum of products from k = row to k = col using read-only memory access (__ldg) for efficiency.

__device__ __forceinline__ float compute_upper_triangular_dot(const float* __restrict__ A,
                                                                const float* __restrict__ B,
                                                                int row, int col, int N) {
    float sum = 0.0f;
    for (int k = row; k <= col; ++k) {
        // Use __ldg to leverage the read-only cache for global memory accesses
        float a_val = __ldg(&A[row * N + k]);
        float b_val = __ldg(&B[k * N + col]);
        sum += a_val * b_val;
    }
    return sum;
}

// Kernel that computes the upper triangular matrix multiplication (C = A * B)
// Only elements where row <= col are computed. The inner computation is refactored
// into a modular device function for better readability and maintainability.

__global__ void modular_upper_triangular_kernel(const float* __restrict__ A,
                                                  const float* __restrict__ B,
                                                  float* __restrict__ C,
                                                  int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N && row <= col) {
        C[row * N + col] = compute_upper_triangular_dot(A, B, row, col, N);
    }
}

// Host function to allocate output tensor, set up grid and block dimensions, and launch the kernel.
// Exposed via pybind11. Uses the same module name TORCH_EXTENSION_NAME as the reference implementation.

torch::Tensor modular_upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    modular_upper_triangular_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &modular_upper_triangular_matmul, "Modular upper triangular matrix multiplication using device functions");
}
