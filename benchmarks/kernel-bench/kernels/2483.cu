#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Configurable block size as template parameter for compile-time optimization
template<int BLOCK_SIZE = 32>
struct SharedMemoryTile {
    template <typename scalar_t>
    __device__ __forceinline__ static void loadA(
        scalar_t (&tileA)[BLOCK_SIZE][BLOCK_SIZE],
        const scalar_t* __restrict__ A,
        const int row,
        const int tile_idx,
        const int M,
        const int K) {
        const int k_index = tile_idx * BLOCK_SIZE + threadIdx.y;
        if (k_index < K && row < M) {
            tileA[threadIdx.y][threadIdx.x] = A[k_index * M + row];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0;
        }
    }

    template <typename scalar_t>
    __device__ __forceinline__ static void loadB(
        scalar_t (&tileB)[BLOCK_SIZE][BLOCK_SIZE],
        const scalar_t* __restrict__ B,
        const int col,
        const int tile_idx,
        const int N,
        const int K) {
        const int k_index = tile_idx * BLOCK_SIZE + threadIdx.x;
        if (k_index < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[col * K + k_index];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0;
        }
    }

    template <typename scalar_t>
    __device__ __forceinline__ static scalar_t computeTileProduct(
        const scalar_t (&tileA)[BLOCK_SIZE][BLOCK_SIZE],
        const scalar_t (&tileB)[BLOCK_SIZE][BLOCK_SIZE]) {
        scalar_t sum = 0;
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum = __fmaf_rn(tileA[k][threadIdx.x], tileB[threadIdx.y][k], sum);
        }
        return sum;
    }
};

template <typename scalar_t, int BLOCK_SIZE = 32>
__global__ void matmul_transpose_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    
    const int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int col = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    __shared__ scalar_t tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ scalar_t tileB[BLOCK_SIZE][BLOCK_SIZE];

    scalar_t sum = 0;
    
    #pragma unroll 4
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        SharedMemoryTile<BLOCK_SIZE>::loadA(tileA, A, row, t, M, K);
        SharedMemoryTile<BLOCK_SIZE>::loadB(tileB, B, col, t, N, K);
        
        __syncthreads();
        
        sum += SharedMemoryTile<BLOCK_SIZE>::computeTileProduct(tileA, tileB);
        
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

    constexpr int BLOCK_SIZE = 32;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_kernel", ([&] {
        matmul_transpose_kernel<scalar_t, BLOCK_SIZE><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Optimized matrix multiplication with transpose (CUDA)");
}