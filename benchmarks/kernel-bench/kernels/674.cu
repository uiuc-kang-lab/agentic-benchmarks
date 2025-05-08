#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define SHARED_PAD 1  // Padding to avoid shared memory bank conflicts

template <typename scalar_t>
__global__ void matmul_optimized_kernel(const scalar_t* __restrict__ A,
                                      const scalar_t* __restrict__ B,
                                      scalar_t* __restrict__ C,
                                      int M, int K, int N) {
    __shared__ scalar_t sA[TILE_SIZE][TILE_SIZE + SHARED_PAD];
    __shared__ scalar_t sB[TILE_SIZE][TILE_SIZE + SHARED_PAD];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    scalar_t acc = 0;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles with coalesced accesses
        int loadA_row = row;
        int loadA_col = t * TILE_SIZE + threadIdx.x;
        if (loadA_row < M && loadA_col < K) {
            sA[threadIdx.y][threadIdx.x] = A[loadA_row * K + loadA_col];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0;
        }

        int loadB_row = t * TILE_SIZE + threadIdx.y;
        int loadB_col = col;
        if (loadB_row < K && loadB_col < N) {
            sB[threadIdx.y][threadIdx.x] = B[loadB_row * N + loadB_col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Unrolled matrix multiply
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "Matrix dimensions mismatch");

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_optimized", [&] {
        matmul_optimized_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    });

    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Optimized matrix multiplication (CUDA)");
}