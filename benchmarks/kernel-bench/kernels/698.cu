#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

template <typename scalar_t>
__global__ void matmul_double_buffered(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B,
                                      scalar_t* __restrict__ C, int M, int K, int N) {
    __shared__ scalar_t sA[2][TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t sB[2][TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    scalar_t accum = 0;

    // Preload first tile
    int load_idx = 0;
    int t = 0;
    if (row < M && t * TILE_WIDTH + threadIdx.x < K)
        sA[load_idx][threadIdx.y][threadIdx.x] = A[row * K + t * TILE_WIDTH + threadIdx.x];
    else
        sA[load_idx][threadIdx.y][threadIdx.x] = 0;

    if (col < N && t * TILE_WIDTH + threadIdx.y < K)
        sB[load_idx][threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * N + col];
    else
        sB[load_idx][threadIdx.y][threadIdx.x] = 0;

    __syncthreads();

    for (t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH - 1; ++t) {
        int compute_idx = load_idx;
        load_idx = 1 - load_idx;

        // Asynchronous load next tile
        if (row < M && (t + 1) * TILE_WIDTH + threadIdx.x < K)
            sA[load_idx][threadIdx.y][threadIdx.x] = A[row * K + (t + 1) * TILE_WIDTH + threadIdx.x];
        else
            sA[load_idx][threadIdx.y][threadIdx.x] = 0;

        if (col < N && (t + 1) * TILE_WIDTH + threadIdx.y < K)
            sB[load_idx][threadIdx.y][threadIdx.x] = B[((t + 1) * TILE_WIDTH + threadIdx.y) * N + col];
        else
            sB[load_idx][threadIdx.y][threadIdx.x] = 0;

        // Compute current tile
        for (int i = 0; i < TILE_WIDTH; ++i) {
            accum += sA[compute_idx][threadIdx.y][i] * sB[compute_idx][i][threadIdx.x];
        }

        __syncthreads();
    }

    // Process last tile
    for (int i = 0; i < TILE_WIDTH; ++i) {
        accum += sA[load_idx][threadIdx.y][i] * sB[load_idx][i][threadIdx.x];
    }

    if (row < M && col < N) {
        C[row * N + col] = accum;
    }
}

torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions must match");

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_double_buffered", ([&] {
        matmul_double_buffered<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Double buffered matrix multiplication");
}