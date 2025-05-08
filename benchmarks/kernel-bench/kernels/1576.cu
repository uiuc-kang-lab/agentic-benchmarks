#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// Modular device functions for smaller tasks within the kernel, optimizing
// the readability, reusability, and maintainability of the code.

__device__ float compute_sum(const float* __restrict__ A, const float* __restrict__ B, int row, int col, int N) {
    float sum = 0.0f;
    for (int k = row; k <= col; ++k) {
        float a_val = __ldg(&A[row * N + k]);
        float b_val = __ldg(&B[k * N + col]);
        sum += a_val * b_val;
    }
    return sum;
}

__global__ void modular_upper_triangular_kernel(const float* __restrict__ A,
                                                 const float* __restrict__ B,
                                                 float* __restrict__ C,
                                                 int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N && row <= col) {
        float sum = compute_sum(A, B, row, col, N);
        C[row * N + col] = sum;
    }
}

// Host function, exposed via pybind11, that wraps the kernel invocation
// It creates a zero tensor for C, launches the kernel, and returns C.

torch::Tensor modular_optimized_upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    // Experimenting with different block sizes to find the optimal configuration
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    modular_upper_triangular_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &modular_optimized_upper_triangular_matmul, "Modular optimized upper triangular matrix multiplication");
}