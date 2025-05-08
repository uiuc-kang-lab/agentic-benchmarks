#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_DIM 16

__global__ void matmul_optimized_indexing_kernel(float const* __restrict__ A, float const* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TILE_DIM][TILE_DIM+1];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        if (row < M && t * TILE_DIM + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_DIM + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && t * TILE_DIM + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum = __fmaf_rn(As[threadIdx.y][k], Bs[k][threadIdx.x], sum);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    torch::Tensor C = torch::zeros({M, N}, options);

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    matmul_optimized_indexing_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication (CUDA)");
}