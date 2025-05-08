#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

template <typename scalar_t>
__device__ inline void load_tile_A(const scalar_t* A, scalar_t* sA_row, int row, int t_col, int M, int K) {
    if (row < M && t_col < K) {
        *sA_row = A[row * K + t_col];
    } else {
        *sA_row = 0;
    }
}

template <typename scalar_t>
__device__ inline void load_tile_B(const scalar_t* B, scalar_t* sB_col, int t_row, int col, int K, int N) {
    if (t_row < K && col < N) {
        *sB_col = B[t_row * N + col];
    } else {
        *sB_col = 0;
    }
}

template <typename scalar_t>
__global__ void modular_matmul_kernel(const scalar_t* __restrict__ A,
                                      const scalar_t* __restrict__ B,
                                      scalar_t* __restrict__ C,
                                      int M, int K, int N) {
    // Pad shared memory to avoid bank conflicts
    // Double buffer shared memory to overlap computation and memory access
    __shared__ scalar_t sA[2][TILE_WIDTH][TILE_WIDTH + 1];
    __shared__ scalar_t sB[2][TILE_WIDTH][TILE_WIDTH + 1];

    const int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    const int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    scalar_t accum = 0;

    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // Calculate tile positions
        const int t_col = t * TILE_WIDTH + threadIdx.x;
        const int t_row = t * TILE_WIDTH + threadIdx.y;

        // Cooperative loading using modular functions
        load_tile_A(A, &sA[threadIdx.y][threadIdx.x], row, t_col, M, K);
        load_tile_B(B, &sB[threadIdx.y][threadIdx.x], t_row, col, K, N);

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_WIDTH; ++k) {
            accum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = accum;
    }
}

torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions must match");

    auto C = torch::empty({M, N}, A.options());

    const dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH,
                     (M + TILE_WIDTH - 1) / TILE_WIDTH);
    const dim3 threads(TILE_WIDTH, TILE_WIDTH);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "modular_matmul", ([&] {
        modular_matmul_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Modular Tiled Matrix Multiplication (CUDA)");
}