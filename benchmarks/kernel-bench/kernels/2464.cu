#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ void load_tile_A(
    scalar_t (&tileA)[16][16],
    const scalar_t* __restrict__ A,
    const int row,
    const int tile_idx,
    const int M,
    const int K) {
    if (tile_idx * 16 + threadIdx.y < K && row < M) {
        tileA[threadIdx.y][threadIdx.x] = A[(tile_idx * 16 + threadIdx.y) * M + row];
    } else {
        tileA[threadIdx.y][threadIdx.x] = 0.0;
    }
}

template <typename scalar_t>
__device__ __forceinline__ void load_tile_B(
    scalar_t (&tileB)[16][16],
    const scalar_t* __restrict__ B,
    const int col,
    const int tile_idx,
    const int N,
    const int K) {
    if (tile_idx * 16 + threadIdx.x < K && col < N) {
        tileB[threadIdx.y][threadIdx.x] = B[col * K + tile_idx * 16 + threadIdx.x];
    } else {
        tileB[threadIdx.y][threadIdx.x] = 0.0;
    }
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_tile_sum(
    const scalar_t (&tileA)[16][16],
    const scalar_t (&tileB)[16][16]) {
    scalar_t sum = 0;
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        sum += tileA[k][threadIdx.x] * tileB[threadIdx.y][k];
    }
    return sum;
}

template <typename scalar_t>
__global__ void matmul_transpose_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ scalar_t tileA[16][16];
    __shared__ scalar_t tileB[16][16];

    scalar_t sum = 0;
    const int num_tiles = (K + 16 - 1) / 16;

    for (int t = 0; t < num_tiles; ++t) {
        load_tile_A(tileA, A, row, t, M, K);
        load_tile_B(tileB, B, col, t, N, K);
        
        __syncthreads();
        
        sum += compute_tile_sum(tileA, tileB);
        
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    const int BLOCK_SIZE = 16;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_kernel", ([&] {
        matmul_transpose_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Matrix multiplication with transpose forward (CUDA)");
}