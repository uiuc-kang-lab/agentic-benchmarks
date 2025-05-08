#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Macro to define the tile/block size. Experiment with different values: 32, 64, 128, etc.
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

// CUDA kernel for matrix multiplication with a dynamic block size
template <typename scalar_t>
__global__ void matmul_dynamic_kernel(const scalar_t* __restrict__ A,
                                        const scalar_t* __restrict__ B,
                                        scalar_t* __restrict__ C,
                                        int M, int K, int N) {
    // Compute global row and column indices
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    __shared__ scalar_t shA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ scalar_t shB[BLOCK_SIZE][BLOCK_SIZE];

    scalar_t value = 0;
    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int tiledA_col = t * BLOCK_SIZE + threadIdx.x;
        int tiledB_row = t * BLOCK_SIZE + threadIdx.y;

        // Load tile from A into shared memory using __ldg for read-only caching
        if (row < M && tiledA_col < K)
            shA[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + tiledA_col]);
        else
            shA[threadIdx.y][threadIdx.x] = 0;

        // Load tile from B into shared memory using __ldg for read-only caching
        if (tiledB_row < K && col < N)
            shB[threadIdx.y][threadIdx.x] = __ldg(&B[tiledB_row * N + col]);
        else
            shB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            value += shA[threadIdx.y][i] * shB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = value;
}

// Host function exposed to Python via Pybind11
torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Tensor B must be a CUDA tensor");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions must match");

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_dynamic_kernel", ([&] {
        matmul_dynamic_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Configurable block size matrix multiplication (CUDA)");
}
