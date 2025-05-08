#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel to compute C = tril(A * B) for lower triangular matrices A and B
// with even workload distribution by allocating a larger tile for each thread block
__global__ void triangular_mm_kernel_even(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           int N) {
    int row_start = blockIdx.y * blockDim.y * 2 + threadIdx.y;
    int row_end = min(row_start + blockDim.y * 2, N);
    int col_start = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int col_end = min(col_start + blockDim.x * 2, N);

    for (int row = row_start; row < row_end; row += blockDim.y) {
        for (int col = col_start; col < col_end; col += blockDim.x) {
            if (row < col) {
                C[row * N + col] = 0.f;
            } else {
                float sum = 0.f;
                for (int k = col; k <= row; ++k) {
                    sum += A[row * N + k] * B[k * N + col];
                }
                C[row * N + col] = sum;
            }
        }
    }
}

// C++ interface exposed to PyTorch
at::Tensor forward_even(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    const int N = A.size(0);
    auto C = torch::empty_like(A);

    const int threads = 16;
    dim3 threadsPerBlock(threads, threads);
    dim3 numBlocks((N + threads * 2 - 1) / (threads * 2), (N + threads * 2 - 1) / (threads * 2));

    triangular_mm_kernel_even<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward_even, "Even-workload triangular matrix multiplication (CUDA)");
}
