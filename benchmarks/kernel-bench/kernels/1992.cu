#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel implementing lower triangular matrix multiplication using stride loops.
// Each thread processes multiple elements using their computed starting index and strides over rows and columns.

__global__ void triangular_mm_stride_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    // Total stride in the row and column dimensions
    int row_stride = blockDim.y * gridDim.y;
    int col_stride = blockDim.x * gridDim.x;

    // Start indices for current thread in the grid
    int start_row = blockIdx.y * blockDim.y + threadIdx.y;
    int start_col = blockIdx.x * blockDim.x + threadIdx.x;

    // Loop over all rows and columns using the computed strides
    for (int row = start_row; row < N; row += row_stride) {
        for (int col = start_col; col < N; col += col_stride) {
            // Only compute for lower triangular part
            if (row < col) {
                C[row * N + col] = 0.0f;
            } else {
                float sum = 0.0f;
                // Compute dot-product for the lower triangular part while handling boundary via stride loops in k if needed
                for (int k = col; k <= row; ++k) {
                    sum += A[row * N + k] * B[k * N + col];
                }
                C[row * N + col] = sum;
            }
        }
    }
}

// C++ interface exposed to PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "A and B must be square matrices");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Block dimensions: choose a configuration that yields enough threads to cover most elements
    const int block_x = 32;
    const int block_y = 16;
    dim3 threads(block_x, block_y);

    // Grid dimensions can be chosen to cover N, but stride loops ensure complete coverage even for larger N
    int grid_x = (N + block_x - 1) / block_x;
    int grid_y = (N + block_y - 1) / block_y;
    dim3 blocks(grid_x, grid_y);

    triangular_mm_stride_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Triangular matrix multiplication with stride loops (CUDA)");
}
