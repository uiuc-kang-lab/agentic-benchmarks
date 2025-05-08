#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>

__global__ void optimized_upper_triangular_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N, int tile_width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tile_col = blockIdx.x * blockDim.x + threadIdx.x;
    int col = tile_col;
    if (row < N && tile_col < N && row <= col) {
        float sum = 0.0f;
        for (int k = row; k <= col; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor optimized_upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);
    
    int tile_width = 256;
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    optimized_upper_triangular_matmul_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, tile_width
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_upper_triangular_matmul, "Optimized upper triangular matrix multiplication");
}