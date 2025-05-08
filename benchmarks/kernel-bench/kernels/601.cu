#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// Device function to load a tile from matrix A into shared memory using __ldg()
template <typename scalar_t>
__device__ inline void load_A_tile(const scalar_t* __restrict__ A,
                                    scalar_t A_tile[TILE_WIDTH][TILE_WIDTH],
                                    int row,
                                    int tile,
                                    int M,
                                    int K) {
    int col = tile * TILE_WIDTH + threadIdx.x;
    if (row < M && col < K)
        A_tile[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + col]);
    else
        A_tile[threadIdx.y][threadIdx.x] = 0;
}

// Device function to load a tile from matrix B into shared memory using __ldg()
template <typename scalar_t>
__device__ inline void load_B_tile(const scalar_t* __restrict__ B,
                                    scalar_t B_tile[TILE_WIDTH][TILE_WIDTH],
                                    int col,
                                    int tile,
                                    int K,
                                    int N) {
    int row = tile * TILE_WIDTH + threadIdx.y;
    if (row < K && col < N)
        B_tile[threadIdx.y][threadIdx.x] = __ldg(&B[row * N + col]);
    else
        B_tile[threadIdx.y][threadIdx.x] = 0;
}

// Device function to compute the dot product for a tile
template <typename scalar_t>
__device__ inline scalar_t compute_tile(const scalar_t A_tile[TILE_WIDTH][TILE_WIDTH],
                                          const scalar_t B_tile[TILE_WIDTH][TILE_WIDTH]) {
    scalar_t sum = 0;
    #pragma unroll
    for (int k = 0; k < TILE_WIDTH; ++k) {
        sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
    }
    return sum;
}

// Modular CUDA kernel for matrix multiplication
template <typename scalar_t>
__global__ void matmul_modular_kernel(const scalar_t* __restrict__ A,
                                        const scalar_t* __restrict__ B,
                                        scalar_t* __restrict__ C,
                                        int M, int K, int N) {
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    scalar_t value = 0;

    __shared__ scalar_t A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t B_tile[TILE_WIDTH][TILE_WIDTH];

    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    
    for (int t = 0; t < numTiles; t++) {
        load_A_tile<scalar_t>(A, A_tile, row, t, M, K);
        load_B_tile<scalar_t>(B, B_tile, col, t, K, N);
        __syncthreads();
        value += compute_tile<scalar_t>(A_tile, B_tile);
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

// Host function exposed to Python via Pybind11
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
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_modular_kernel", ([&] {
        matmul_modular_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N
        );
    }));
    
    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Modular device function matrix multiplication (CUDA)");
}
