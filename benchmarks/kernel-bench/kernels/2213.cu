#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_DIM 16
#define BLOCK_K 32

__global__ void stridedTiledMatMulKernel(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int K, int M, int N) {
    __shared__ float As[TILE_DIM][BLOCK_K];
    __shared__ float Bs[BLOCK_K][TILE_DIM];

    int blockRow = blockIdx.x;
    int blockCol = blockIdx.y;

    int row = threadIdx.x;
    int col = threadIdx.y;

    float Csub = 0.0f;

    // Loop over all tiles
    for (int s = 0; s < (K + BLOCK_K - 1) / BLOCK_K; ++s) {
        if ((s * BLOCK_K + col) < K && (blockRow * TILE_DIM + row) < M) {
            As[row][col] = A[(s * BLOCK_K + col) * M + blockRow * TILE_DIM + row];
        } else {
            As[row][col] = 0.0f;
        }

        if ((s * BLOCK_K + row) < K && (blockCol * TILE_DIM + col) < N) {
            Bs[row][col] = B[(s * BLOCK_K + row) * N + blockCol * TILE_DIM + col];
        } else {
            Bs[row][col] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int e = 0; e < BLOCK_K; ++e) {
            Csub += As[row][e] * Bs[e][col];
        }

        __syncthreads();
    }

    // Accumulate result in global memory
    if (blockRow * TILE_DIM + row < M && blockCol * TILE_DIM + col < N) {
        atomicAdd(&C[(blockRow * TILE_DIM + row) * N + blockCol * TILE_DIM + col], Csub);
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch");
    int N = B.size(1);

    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((M + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    stridedTiledMatMulKernel<<<grid, block>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Strided tiled matrix multiplication (CUDA)");
}
