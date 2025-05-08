#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// Kernel using stride loops to handle workloads exceeding the number of threads
__global__ void stride_upper_triangular_kernel(const float* __restrict__ A,
                                                const float* __restrict__ B,
                                                float* __restrict__ C,
                                                int N) {
    // Compute initial row and column indices for the current thread
    int row_init = blockIdx.y * blockDim.y + threadIdx.y;
    int col_init = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute strides based on grid and block dimensions
    int row_stride = blockDim.y * gridDim.y;
    int col_stride = blockDim.x * gridDim.x;

    // Loop over rows with stride
    for (int row = row_init; row < N; row += row_stride) {
        // Loop over columns with stride
        for (int col = col_init; col < N; col += col_stride) {
            // Process only the upper triangular part
            if (row <= col) {
                float sum = 0.0f;
                // Compute dot product for this (row, col) in the valid range
                for (int k = row; k <= col; ++k) {
                    sum += A[row * N + k] * B[k * N + col];
                }
                C[row * N + col] = sum;
            }
        }
    }
}

// Host function exposed via Pybind11
torch::Tensor stride_upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    // Configure block and grid sizes
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    stride_upper_triangular_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &stride_upper_triangular_matmul, "Stride loop based upper triangular matrix multiplication");
}
