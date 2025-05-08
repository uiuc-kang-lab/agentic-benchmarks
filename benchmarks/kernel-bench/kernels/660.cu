#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

template <typename scalar_t>
__forceinline__ __device__ void load_tile_A(
    scalar_t (&sA)[TILE_WIDTH][TILE_WIDTH],
    const scalar_t* __restrict__ A,
    int row, int tiledCol, int M, int K) {
    if (row < M && tiledCol < K)
        sA[threadIdx.y][threadIdx.x] = A[row * K + tiledCol];
    else
        sA[threadIdx.y][threadIdx.x] = 0;
}

template <typename scalar_t>
__forceinline__ __device__ void load_tile_B(
    scalar_t (&sB)[TILE_WIDTH][TILE_WIDTH],
    const scalar_t* __restrict__ B,
    int tiledRow, int col, int K, int N) {
    if (tiledRow < K && col < N)
        sB[threadIdx.y][threadIdx.x] = B[tiledRow * N + col];
    else
        sB[threadIdx.y][threadIdx.x] = 0;
}

template <typename scalar_t>
__forceinline__ __device__ scalar_t compute_tile_element(
    const scalar_t (&sA)[TILE_WIDTH][TILE_WIDTH],
    const scalar_t (&sB)[TILE_WIDTH][TILE_WIDTH]) {
    scalar_t sum = 0;
    #pragma unroll
    for (int i = 0; i < TILE_WIDTH; i++) {
        sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
    }
    return sum;
}

template <typename scalar_t>
__global__ void matmul_cuda_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int M, int K, int N) {
    
    __shared__ scalar_t sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t sB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    scalar_t sum = 0;

    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        int tiledACol = t * TILE_WIDTH + threadIdx.x;
        int tiledBRow = t * TILE_WIDTH + threadIdx.y;

        // Load tiles using device functions
        load_tile_A(sA, A, row, tiledACol, M, K);
        load_tile_B(sB, B, tiledBRow, col, K, N);

        __syncthreads();

        // Compute partial result for this tile
        sum += compute_tile_element(sA, sB);

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);

    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, 
                (M + TILE_WIDTH - 1) / TILE_WIDTH);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_cuda_kernel", ([&] {
        matmul_cuda_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Modular matrix multiplication forward (CUDA)");
}