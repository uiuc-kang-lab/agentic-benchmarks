#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// CUDA kernel for matrix multiplication with minimized warp divergence
// The kernel distinguishes between interior blocks (fully within matrix bounds) and boundary blocks.
// For interior blocks, bounds checks are omitted to ensure uniform control flow and avoid divergent branching.

template <typename scalar_t>
__global__ void matmul_cuda_kernel(const scalar_t* __restrict__ A,
                                     const scalar_t* __restrict__ B,
                                     scalar_t* __restrict__ C,
                                     int M, int K, int N) {
    // Compute global row and column indices
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    scalar_t value = 0;

    // Determine if this block is completely interior (i.e. no boundary checks needed).
    // If (blockIdx.x+1)*TILE_WIDTH <= N and (blockIdx.y+1)*TILE_WIDTH <= M, the entire tile is within bounds.
    bool interior_block = ((blockIdx.x + 1) * TILE_WIDTH <= N) && ((blockIdx.y + 1) * TILE_WIDTH <= M);

    __shared__ scalar_t sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t sB[TILE_WIDTH][TILE_WIDTH];

    int num_tiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int t = 0; t < num_tiles; ++t) {
        int tiledA_col = t * TILE_WIDTH + threadIdx.x;
        int tiledB_row = t * TILE_WIDTH + threadIdx.y;

        // For interior blocks, all accesses are within bounds so we avoid conditional checks.
        if (interior_block) {
            sA[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + tiledA_col]);
            sB[threadIdx.y][threadIdx.x] = __ldg(&B[tiledB_row * N + col]);
        } else {
            // For boundary blocks, use a ternary operator for a branchless-like selection
            sA[threadIdx.y][threadIdx.x] = (row < M && tiledA_col < K) ? __ldg(&A[row * K + tiledA_col]) : static_cast<scalar_t>(0);
            sB[threadIdx.y][threadIdx.x] = (col < N && tiledB_row < K) ? __ldg(&B[tiledB_row * N + col]) : static_cast<scalar_t>(0);
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; ++i) {
            value += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        __syncthreads();
    }

    // Write the computed result if within output bounds
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

// Forward function called from Python
torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_cuda_kernel", ([&] {
        matmul_cuda_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &module_fn, "Matrix multiplication with minimized warp divergence");
}
