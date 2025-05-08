#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// Optimized CUDA kernel for matrix multiplication using __ldg() for read-only global memory loads
// and assuming that the input matrices A and B are allocated on 128-bit boundaries.

template <typename scalar_t>
__global__ void matmul_cuda_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B,
                                     scalar_t* __restrict__ C, int M, int K, int N) {
    __shared__ scalar_t sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t sB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y; // Index in M dimension
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x; // Index in N dimension

    scalar_t value = 0;
    int num_tiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < num_tiles; ++t) {
        int tiledA_col = t * TILE_WIDTH + threadIdx.x;
        int tiledB_row = t * TILE_WIDTH + threadIdx.y;

        // Use __ldg() for read-only accesses, assuming A and B are 128-bit aligned
        if (row < M && tiledA_col < K)
            sA[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + tiledA_col]);
        else
            sA[threadIdx.y][threadIdx.x] = 0;

        if (col < N && tiledB_row < K)
            sB[threadIdx.y][threadIdx.x] = __ldg(&B[tiledB_row * N + col]);
        else
            sB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; ++i) {
            value += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

// Forward function

torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    // Allocate output tensor
    auto C = torch::empty({M, N}, A.options());

    dim3 threads_per_block(TILE_WIDTH, TILE_WIDTH);
    dim3 num_blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_cuda_kernel", ([&] {
        matmul_cuda_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    cudaDeviceSynchronize();
    return C;
}

// Pybind11 module binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Optimized matrix multiplication forward (CUDA, __ldg and 128-bit aligned loads)");
}
