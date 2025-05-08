#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel with block size experimentation for better performance
__global__ void optimized_block_size_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= N || col >= N) return;

    if (row < col) {
        C[row * N + col] = 0.0f;
    } else {
        float sum = 0.0f;
        for (int k = col; k <= row; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    const int N = A.size(0);
    auto C = torch::empty_like(A);

    // Experiment with block size 64x16 for optimal performance
    const int bx = 64, by = 16;
    dim3 threads(bx, by);
    dim3 blocks((N + bx - 1) / bx, (N + by - 1) / by);

    optimized_block_size_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Optimized block size triangular matmul (CUDA)");
}
