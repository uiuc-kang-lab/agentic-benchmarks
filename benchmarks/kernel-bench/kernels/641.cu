#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Using a moderate tile width
#define TILE_WIDTH 16

// CUDA kernel for matrix multiplication using shared memory tiling
// This version minimizes the use of __syncthreads() by only synchronizing after loading shared memory
// and avoiding an extra synchronization at the end of the loop iteration.

template <typename scalar_t>
__global__ void matmul_less_sync_kernel(const scalar_t* __restrict__ A,
                                          const scalar_t* __restrict__ B,
                                          scalar_t* __restrict__ C,
                                          int M, int K, int N) {
    // Declare shared memory for tiles of A and B
    __shared__ scalar_t sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t sB[TILE_WIDTH][TILE_WIDTH];

    // Calculate row and column index of C element computed by this thread
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    scalar_t sum = 0;
    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t++) {
        // Load a tile of A into shared memory
        int tiledACol = t * TILE_WIDTH + threadIdx.x;
        if (row < M && tiledACol < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + tiledACol];
        else
            sA[threadIdx.y][threadIdx.x] = scalar_t(0);

        // Load a tile of B into shared memory
        int tiledBRow = t * TILE_WIDTH + threadIdx.y;
        if (tiledBRow < K && col < N)
            sB[threadIdx.y][threadIdx.x] = B[tiledBRow * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = scalar_t(0);

        // Synchronize threads to ensure the tile is fully loaded
        __syncthreads();

        // Compute partial results using the loaded tile
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i++) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        // For all iterations except the last one, synchronize threads to avoid race conditions
        // when loading new data into shared memory.
        if (t != numTiles - 1) {
            __syncthreads();
        }
    }

    // Write the computed value to the output matrix if within bounds
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Forward function: performs necessary checks, sets up dimensions, and launches the kernel
torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    auto C = torch::empty({M, N}, A.options());

    // Define block and grid dimensions
    dim3 threads_per_block(TILE_WIDTH, TILE_WIDTH);
    dim3 num_blocks((N + TILE_WIDTH - 1) / TILE_WIDTH,
                     (M + TILE_WIDTH - 1) / TILE_WIDTH);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_less_sync_kernel", ([&] {
        matmul_less_sync_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    cudaDeviceSynchronize();
    return C;
}

// Binding code using pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Matrix multiplication forward (CUDA) with fewer synchronizations");
}
