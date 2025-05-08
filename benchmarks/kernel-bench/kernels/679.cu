#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32
#define PAD 1
#define UNROLL_FACTOR 4

template <typename scalar_t>
__global__ void matmul_shared_vec4_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B,
                                        scalar_t* __restrict__ C, int M, int K, int N) {
    __shared__ scalar_t sA[TILE_WIDTH][TILE_WIDTH + PAD];
    __shared__ scalar_t sB[TILE_WIDTH][TILE_WIDTH + PAD];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    scalar_t tmp_val = 0;

    for (int t = 0; t < K; t += TILE_WIDTH) {
        int tiled_t = t + threadIdx.x;
        if (row < M && tiled_t < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + tiled_t];
        else
            sA[threadIdx.y][threadIdx.x] = 0;

        tiled_t = t + threadIdx.y;
        if (col < N && tiled_t < K)
            sB[threadIdx.y][threadIdx.x] = B[tiled_t * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        #pragma unroll 4
        for (int k = 0; k < TILE_WIDTH; k++) {
            tmp_val += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = tmp_val;
}

torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors.");
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_shared_vec4_kernel", [&] {
        matmul_shared_vec4_kernel<scalar_t><<<grid, block>>>(
            A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(), C.data_ptr<scalar_t>(), M, K, N);
    });

    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Matrix multiplication with shared mem tiling and vector loads");
}