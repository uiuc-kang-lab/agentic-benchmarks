#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// Device function to load a tile from matrix A into shared memory using __ldg
template <typename scalar_t>
__device__ void load_A_tile(const scalar_t* __restrict__ A, scalar_t sA[TILE_WIDTH][TILE_WIDTH], int M, int K, int t) {
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = t * TILE_WIDTH + threadIdx.x;
    if (row < M && col < K)
        sA[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + col]);
    else
        sA[threadIdx.y][threadIdx.x] = 0;
}

// Device function to load a tile from matrix B into shared memory using __ldg
template <typename scalar_t>
__device__ void load_B_tile(const scalar_t* __restrict__ B, scalar_t sB[TILE_WIDTH][TILE_WIDTH], int N, int K, int t) {
    int row = t * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    if (row < K && col < N)
        sB[threadIdx.y][threadIdx.x] = __ldg(&B[row * N + col]);
    else
        sB[threadIdx.y][threadIdx.x] = 0;
}

// Device function to compute partial product for the loaded tiles
template <typename scalar_t>
__device__ scalar_t compute_tile(const scalar_t sA[TILE_WIDTH][TILE_WIDTH], const scalar_t sB[TILE_WIDTH][TILE_WIDTH]) {
    scalar_t sum = 0;
    #pragma unroll
    for (int i = 0; i < TILE_WIDTH; ++i) {
        sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
    }
    return sum;
}

// Modular CUDA kernel for matrix multiplication using shared memory tiling and modular device functions
template <typename scalar_t>
__global__ void matmul_cuda_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B,
                                     scalar_t* __restrict__ C, int M, int K, int N) {
    __shared__ scalar_t sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t sB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    scalar_t value = 0;
    int num_tiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < num_tiles; ++t) {
        // Load the current tile for A and B using modular device functions
        load_A_tile<scalar_t>(A, sA, M, K, t);
        load_B_tile<scalar_t>(B, sB, N, K, t);
        __syncthreads();

        // Compute the partial product for the current tile
        value += compute_tile<scalar_t>(sA, sB);
        __syncthreads();
    }

    // Write the result to global memory
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

// Forward function called by the Pybind11 module
torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

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
    m.def("forward", &module_fn, "Modular matrix multiplication forward (CUDA, modular device functions)");
}
