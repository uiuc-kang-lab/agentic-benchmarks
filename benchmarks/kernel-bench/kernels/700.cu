#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// CUDA kernel for matrix multiplication using shared memory tiling with loop unrolling
template <typename scalar_t>
__global__ void matmul_unroll_kernel(const scalar_t* __restrict__ A,
                                       const scalar_t* __restrict__ B,
                                       scalar_t* __restrict__ C,
                                       int M, int K, int N) {
    __shared__ scalar_t tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t tileB[TILE_WIDTH][TILE_WIDTH];

    // Compute global row and column indices
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    scalar_t value = 0;

    // Calculate the number of tiles
    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    // Loop over the tiles
    for (int t = 0; t < numTiles; t++) {
        int colA = t * TILE_WIDTH + threadIdx.x;
        int rowB = t * TILE_WIDTH + threadIdx.y;

        // Load tile from A into shared memory
        tileA[threadIdx.y][threadIdx.x] = (row < M && colA < K) ? A[row * K + colA] : 0; __syncthreads();
        // Load tile from B into shared memory
        tileB[threadIdx.y][threadIdx.x] = (rowB < K && col < N) ? B[rowB * N + col] : 0;

        __syncthreads();

        // Compute partial dot product with loop unrolling
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i++) {
            value += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result to global memory
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

// Forward function
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

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_unroll_kernel", ([&] {
        matmul_unroll_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    cudaDeviceSynchronize();
    return C;
}

// Binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Matrix multiplication forward (CUDA) with loop unrolling");
}
