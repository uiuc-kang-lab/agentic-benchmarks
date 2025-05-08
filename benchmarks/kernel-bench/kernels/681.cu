#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

// This kernel uses shared memory tiling and leverages __ldg() to perform read-only loads from global memory.
// It assumes that the input matrices A and B are aligned to 128-bit boundaries, which enables optimal vectorized loads under the hood.

template <typename scalar_t>
__global__ void ldg_matmul_kernel(const scalar_t* __restrict__ A,
                                   const scalar_t* __restrict__ B,
                                   scalar_t* __restrict__ C,
                                   int M, int K, int N) {
    __shared__ scalar_t sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t sB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    scalar_t value = 0;

    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_WIDTH + threadIdx.x;
        int bRow = t * TILE_WIDTH + threadIdx.y;

        // Use __ldg() for read-only global memory loads. This benefits from the read-only cache
        // when the data is 128-bit aligned.
        if (row < M && aCol < K)
            sA[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + aCol]);
        else
            sA[threadIdx.y][threadIdx.x] = 0;

        if (bRow < K && col < N)
            sB[threadIdx.y][threadIdx.x] = __ldg(&B[bRow * N + col]);
        else
            sB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i++) {
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
    TORCH_CHECK(A.is_cuda(), "Tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Tensor B must be a CUDA tensor");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "ldg_matmul_kernel", ([&] {
        ldg_matmul_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    cudaDeviceSynchronize();
    return C;
}

// Binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "LDG optimized matrix multiplication (CUDA)");
}
