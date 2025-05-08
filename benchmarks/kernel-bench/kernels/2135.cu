#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to compute dot product for a specific row/col pair
__device__ float compute_triangular_dot_product(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               const int row,
                                               const int col,
                                               const int N) {
    float sum = 0.f;
    // Only compute from col to row for lower triangular matrices
    #pragma unroll
    for (int k = col; k <= row; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    return sum;
}

// CUDA kernel to compute C = tril(A * B) for lower triangular matrices A and B
__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    const int N) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Early exit conditions
    if (row >= N || col >= N) return;
    if (row < col) {
        C[row * N + col] = 0.f;
        return;
    }

    // Compute result using the device function
    C[row * N + col] = compute_triangular_dot_product(A, B, row, col, N);
}

// C++ interface exposed to PyTorch
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

    // Optimize block size for H100
    const int threads = 32;
    dim3 threadsPerBlock(threads, threads);
    dim3 numBlocks((N + threads - 1) / threads, (N + threads - 1) / threads);

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
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}