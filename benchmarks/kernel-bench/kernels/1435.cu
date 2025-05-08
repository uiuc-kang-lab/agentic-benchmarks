#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void matmul_kernel_warp(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;

    const int row = blockIdx.y * TILE_SIZE + warp_id;
    const int col = blockIdx.x * TILE_SIZE + lane_id;

    float value = 0.0f;

    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        float a_val = 0.0f, b_val = 0.0f;

        if (row < N && (m * TILE_SIZE + lane_id) < N)
            a_val = A[row * N + m * TILE_SIZE + lane_id];

        if ((m * TILE_SIZE + warp_id) < N && col < N)
            b_val = B[(m * TILE_SIZE + warp_id) * N + col];

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            float a = __shfl_sync(0xFFFFFFFF, a_val, k);
            float b = __shfl_sync(0xFFFFFFFF, b_val, k);
            value += a * b;
        }
    }

    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of same size");

    const int N = A.size(0);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    const int threads_per_block = TILE_SIZE * TILE_SIZE;
    dim3 threads(threads_per_block);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel_warp<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix Multiplication (CUDA)");
}