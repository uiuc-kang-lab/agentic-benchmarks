#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#define TILE_M 16
#define TILE_N 16
#define TILE_K 32
#define WARPS_PER_BLOCK 4
#define THREADS_PER_WARP 32

using namespace cooperative_groups;

__global__ void optimizedMatmulKernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int K, int M, int N) {
    __shared__ float Ashared[TILE_K][TILE_M + 1];
    __shared__ float Bshared[TILE_K][TILE_N + 1];

    int warpId = threadIdx.x / THREADS_PER_WARP;
    int laneId = threadIdx.x % THREADS_PER_WARP;

    int row = blockIdx.y * TILE_M + (warpId / (TILE_N / 16)) * 16 + (laneId / 4);
    int col = blockIdx.x * TILE_N + (warpId % (TILE_N / 16)) * 16 + (laneId % 4);

    float accum[TILE_M/TILE_N] = {0.0f};

    for (int k_base = 0; k_base < K; k_base += TILE_K) {
        // Load A tile (transposed access pattern)
        int loadA_row = k_base + threadIdx.x % TILE_K;
        int loadA_col = blockIdx.y * TILE_M + threadIdx.x / TILE_K;
        if (loadA_row < K && loadA_col < M)
            Ashared[threadIdx.x % TILE_K][threadIdx.x / TILE_K] = A[loadA_row * M + loadA_col];

        // Load B tile
        int loadB_row = k_base + threadIdx.x % TILE_K;
        int loadB_col = blockIdx.x * TILE_N + threadIdx.x / TILE_K;
        if (loadB_row < K && loadB_col < N)
            Bshared[threadIdx.x % TILE_K][threadIdx.x / TILE_K] = B[loadB_row * N + loadB_col];

        __syncthreads();

        // Compute partial sums
        for (int k = 0; k < TILE_K; ++k) {
            float a = Ashared[k][threadIdx.y];
            float b = Bshared[k][threadIdx.z];
            accum[0] += a * b;
        }

        __syncthreads();
    }

    // Warp-level reduction using shfl
    for (int offset = 16; offset > 0; offset /= 2) {
        accum[0] += __shfl_down_sync(0xffffffff, accum[0], offset);
    }

    if (laneId == 0 && row < M && col < N) {
        C[row * N + col] = accum[0];
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");

    int K = A.size(0);
    int M = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch");

    auto C = torch::zeros({M, N}, A.options());

    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    dim3 block(THREADS_PER_WARP * WARPS_PER_BLOCK, TILE_M / 16, TILE_N / 16);

    optimizedMatmulKernel<<<grid, block>>>(A.data_ptr<float>(),
                                         B.data_ptr<float>(),
                                         C.data_ptr<float>(),
                                         K, M, N);

    cudaDeviceSynchronize();
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized transposed matrix multiplication");
}
