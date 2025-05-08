#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// Unified device function to load tiles from matrix A and B into shared memory using __ldg
// This reduces code duplication and enhances readability
template <typename scalar_t>
__device__ void load_tile(const scalar_t* __restrict__ matrix, scalar_t tile[TILE_WIDTH][TILE_WIDTH], int row, int col, int max_row, int max_col, bool isA) {
    if (row < max_row && col < max_col) {
        tile[threadIdx.y][threadIdx.x] = __ldg(&matrix[row * (isA ? max_col : max_row) + col]);
    } else {
        tile[threadIdx.y][threadIdx.x] = 0;
    }
}

// Unified CUDA kernel for matrix multiplication using shared memory tiling and modular device functions
// Combines the benefits of both kernels for enhanced performance
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
        // Load the current tile for A and B using unified device function
        load_tile<scalar_t>(A, sA, row, t * TILE_WIDTH + threadIdx.x, M, K, true);
        load_tile<scalar_t>(B, sB, t * TILE_WIDTH + threadIdx.y, col, K, N, false);
        __syncthreads();

        // Compute the partial product for the current tile
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; ++i) {
            value += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
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
    m.def("forward", &module_fn, "Optimized modular matrix multiplication forward (CUDA)");
}