#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

__device__ float compute_element(const float* A, const float* B, int row, int col, int N) {
    float sum = 0.0f;
    for (int k = row; k <= col; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    return sum;
}

__global__ void upper_triangular_matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N && row <= col) {
        C[row * N + col] = compute_element(A, B, row, col, N);
    }
}

torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);

    auto C = torch::zeros_like(A);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    upper_triangular_matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Upper triangular matrix multiplication");
}