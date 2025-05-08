#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>

// Optimized kernel that combines the benefits of both balanced workload distribution and pipelining
// for upper triangular matrix multiplication. It uses a tile-based approach with streams to overlap
// computation and data transfer, while ensuring efficient use of threads.
__global__ void optimized_upper_triangular_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int col_offset,
    int tile_width
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tile_col = blockIdx.x * blockDim.x + threadIdx.x;
    int col = col_offset + tile_col;
    if (row < N && tile_col < tile_width) {
        float sum = 0.0f;
        if (row <= col) {
            for (int k = row; k <= col; k++) {
                sum += __ldg(&A[row * N + k]) * __ldg(&B[k * tile_width + tile_col]);
            }
        }
        C[row * tile_width + tile_col] = sum;
    }
}

// Host function that combines pipelining and optimized kernel execution
// for upper triangular matrix multiplication.
torch::Tensor optimized_upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    int tile_width = 256;
    int num_tiles = (N + tile_width - 1) / tile_width;

    auto C = torch::empty({N, N}, A.options());

    float* d_A = nullptr;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMemcpy(d_A, A.data_ptr<float>(), N * N * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    float* d_B_tile[2];
    float* d_C_tile[2];
    for (int i = 0; i < 2; i++) {
        cudaMalloc(&d_B_tile[i], N * tile_width * sizeof(float));
        cudaMalloc(&d_C_tile[i], N * tile_width * sizeof(float));
    }

    dim3 blockDim(32, 32);

    for (int t = 0; t < num_tiles; t++) {
        int current_tile_width = std::min(tile_width, N - t * tile_width);
        int col_offset = t * tile_width;
        dim3 gridDim((current_tile_width + blockDim.x - 1) / blockDim.x,
                     (N + blockDim.y - 1) / blockDim.y);
        cudaStream_t stream = streams[t % 2];

        cudaMemcpy2DAsync(
            d_B_tile[t % 2],
            current_tile_width * sizeof(float),
            B.data_ptr<float>() + col_offset,
            N * sizeof(float),
            current_tile_width * sizeof(float),
            N,
            cudaMemcpyHostToDevice, stream);

        optimized_upper_triangular_kernel<<<gridDim, blockDim, 0, stream>>>(
            d_A, d_B_tile[t % 2], d_C_tile[t % 2],
            N, col_offset, current_tile_width
        );

        cudaMemcpy2DAsync(
            C.data_ptr<float>() + col_offset,
            N * sizeof(float),
            d_C_tile[t % 2],
            current_tile_width * sizeof(float),
            current_tile_width * sizeof(float),
            N,
            cudaMemcpyDeviceToHost, stream);
    }

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    for (int i = 0; i < 2; i++) {
        cudaFree(d_B_tile[i]);
        cudaFree(d_C_tile[i]);
    }
    cudaFree(d_A);
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_upper_triangular_matmul, "Optimized upper triangular matrix multiplication");
}
