#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// Device function to load a tile from matrix A into shared memory using __ldg()
template <typename scalar_t>
__device__ __forceinline__ void load_tile_A(const scalar_t* __restrict__ A,
                                              scalar_t tile_A[TILE_WIDTH][TILE_WIDTH],
                                              int row, int tile_idx, int K, int M) {
    int col = tile_idx * TILE_WIDTH + threadIdx.x;
    tile_A[threadIdx.y][threadIdx.x] = (row < M && col < K) ? __ldg(&A[row * K + col]) : static_cast<scalar_t>(0);
}

// Device function to load a tile from matrix B into shared memory using __ldg()
template <typename scalar_t>
__device__ __forceinline__ void load_tile_B(const scalar_t* __restrict__ B,
                                              scalar_t tile_B[TILE_WIDTH][TILE_WIDTH],
                                              int col, int tile_idx, int K, int N) {
    int row = tile_idx * TILE_WIDTH + threadIdx.y;
    tile_B[threadIdx.y][threadIdx.x] = (row < K && col < N) ? __ldg(&B[row * N + col]) : static_cast<scalar_t>(0);
}

// Device function to compute the partial dot product for the current tile
template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_tile(const scalar_t tile_A[TILE_WIDTH][TILE_WIDTH],
                                                    const scalar_t tile_B[TILE_WIDTH][TILE_WIDTH]) {
    scalar_t partial = 0;
    #pragma unroll
    for (int i = 0; i < TILE_WIDTH; ++i) {
        partial += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
    }
    return partial;
}

// Modular CUDA kernel for matrix multiplication using shared memory tiling
// and modular device functions for loading and computing tiles
template <typename scalar_t>
__global__ void matmul_modular_refactored_kernel(const scalar_t* __restrict__ A,
                                                   const scalar_t* __restrict__ B,
                                                   scalar_t* __restrict__ C,
                                                   int M, int K, int N) {
    __shared__ scalar_t tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t tile_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    scalar_t sum = 0;

    int num_tiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < num_tiles; ++t) {
        load_tile_A<scalar_t>(A, tile_A, row, t, K, M);
        load_tile_B<scalar_t>(B, tile_B, col, t, K, N);
        __syncthreads();

        sum += compute_tile<scalar_t>(tile_A, tile_B);
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host function to launch the CUDA kernel
torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_modular_refactored_kernel", ([&] {
        matmul_modular_refactored_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N
        );
    }));

    cudaDeviceSynchronize();
    return C;
}

// Pybind11 module binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Matrix multiplication using modular device functions (CUDA)");
}
