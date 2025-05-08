#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// This implementation introduces atomic operations only when necessary to handle race conditions.
// Usage is minimized to reduce contention within global memory using fine-grained atomic addition.

template <typename scalar_t>
__global__ void atomic_matmul_kernel(const scalar_t* __restrict__ A,
                                      const scalar_t* __restrict__ B,
                                      scalar_t* __restrict__ C,
                                      int M, int K, int N) {
    __shared__ scalar_t sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t sB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    scalar_t value = 0;

    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int t = 0; t < numTiles; ++t) {
        int tiledACol = t * TILE_WIDTH + threadIdx.x;
        if (row < M && tiledACol < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + tiledACol];
        else
            sA[threadIdx.y][threadIdx.x] = scalar_t(0);

        int tiledBRow = t * TILE_WIDTH + threadIdx.y;
        if (tiledBRow < K && col < N)
            sB[threadIdx.y][threadIdx.x] = B[tiledBRow * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = scalar_t(0);

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; ++i) {
            value += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        if (t != numTiles - 1) {
            __syncthreads();
        }
    }

    if (row < M && col < N) {
        atomicAdd(&C[row * N + col], value); // Using fine-grained atomic operation here
    }
}

torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads_per_block(TILE_WIDTH, TILE_WIDTH);
    dim3 num_blocks((N + TILE_WIDTH - 1) / TILE_WIDTH,
                     (M + TILE_WIDTH - 1) / TILE_WIDTH);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "atomic_matmul_kernel", ([&] {
        atomic_matmul_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Matrix multiplication forward (CUDA) with atomic operations");
}

