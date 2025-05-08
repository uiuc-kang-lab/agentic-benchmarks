#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>

// Optimized kernel to combine pipelining and balanced workload for upper triangular matrix multiplication.
__global__ void optimized_upper_triangular_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int start_col,
    int tile_width
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tile_col = blockIdx.x * blockDim.x + threadIdx.x;
    int col = start_col + tile_col;
    if (row < N && tile_col < tile_width) {
        float sum = 0.0f;
        if (row <= col) {
            for (int k = row; k <= col; ++k) {
                sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
            }
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor optimized_upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);

    // Choose a suitable tile width for partitioning, taking into account GPU capabilities
    int tile_width = 256;
    int num_tiles = (N + tile_width - 1) / tile_width;

    // Allocate the output tensor C and device memory
    auto C = torch::empty({N, N}, A.options());

    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    cudaMemcpy(d_A, A.data_ptr<float>(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data_ptr<float>(), N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);

    for (int t = 0; t < num_tiles; t++) {
        int current_tile_width = std::min(tile_width, N - t * tile_width);
        int start_col = t * tile_width;
        dim3 gridDim((current_tile_width + blockDim.x - 1) / blockDim.x,
                     (N + blockDim.y - 1) / blockDim.y);

        optimized_upper_triangular_kernel<<<gridDim, blockDim>>>(
            d_A, d_B, d_C, N, start_col, current_tile_width
        );
    }

    cudaMemcpy(C.data_ptr<float>(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_upper_triangular_matmul, "Optimized upper triangular matrix multiplication");
}