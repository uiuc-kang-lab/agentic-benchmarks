#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void matmul_kernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int N) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float value = 0;

    for (int i = 0; i < (N + TILE_SIZE - 1) / TILE_SIZE; ++i) {
        if (row < N && i * TILE_SIZE + tx < N)
            s_A[ty][tx] = A[row * N + i * TILE_SIZE + tx];
        else
            s_A[ty][tx] = 0.0f;

        if (col < N && i * TILE_SIZE + ty < N)
            s_B[ty][tx] = B[(i * TILE_SIZE + ty) * N + col];
        else
            s_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            value += s_A[ty][k] * s_B[k][tx];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = value;
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Matrices must be square");

    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Ensure grid dimensions do not exceed maximum
    blocks.x = min(blocks.x, (65536 / threads.x));
    blocks.y = min(blocks.y, (65536 / threads.y));

    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Thread Block Assignment Matrix Multiplication");
}