#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel leverages shared memory with padding to avoid bank conflicts. 
// For each tile over the K dimension, a tile from A and B is loaded into shared memory.
// The extra padding in the shared memory arrays ensures that bank conflicts are minimized,
// which reduces global memory latency. Synchronizations (__syncthreads()) ensure proper allocation
// and avoid race conditions while the inner product for each tile is computed.

template <typename scalar_t>
__global__ void matmul_transpose_shared_pad_kernel(
    const scalar_t* __restrict__ A,  // A: (K x M), stored as A[k * M + m]
    const scalar_t* __restrict__ B,  // B: (N x K), stored as B[n * K + k]
    scalar_t* __restrict__ C,        // C: (M x N), stored as C[m * N + n]
    const int M,                     // M: number of columns in A (and rows in C)
    const int N,                     // N: number of rows in B (and columns in C)
    const int K) {                   // K: common dimension

    // Define tile size 
    const int TILE_SIZE = 16;

    // Shared memory arrays with an extra column for padding to avoid bank conflicts
    __shared__ scalar_t A_shared[TILE_SIZE][TILE_SIZE + 1];
    __shared__ scalar_t B_shared[TILE_SIZE][TILE_SIZE + 1];

    // Each block computes a TILE_SIZE x TILE_SIZE submatrix of C.
    // Map block indices to output matrix indices.
    int row = blockIdx.x * TILE_SIZE + threadIdx.x;  // corresponds to m in C
    int col = blockIdx.y * TILE_SIZE + threadIdx.y;  // corresponds to n in C

    scalar_t acc = 0;

    // Number of tiles along the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Loop over tiles
    for (int t = 0; t < numTiles; t++) {
        // Load a tile of A into shared memory.
        // A is stored as (K x M): A[k, m] = A[k * M + m].
        int a_k = t * TILE_SIZE + threadIdx.y;  // k index for A
        if (row < M && a_k < K) {
            A_shared[threadIdx.y][threadIdx.x] = A[a_k * M + row];
        } else {
            A_shared[threadIdx.y][threadIdx.x] = 0;
        }

        // Load a tile of B into shared memory.
        // B is stored as (N x K): B[n, k] = B[n * K + k].
        int b_k = t * TILE_SIZE + threadIdx.x;  // k index for B
        if (col < N && b_k < K) {
            B_shared[threadIdx.y][threadIdx.x] = B[col * K + b_k];
        } else {
            B_shared[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            acc += A_shared[i][threadIdx.x] * B_shared[threadIdx.y][i];
        }

        __syncthreads();
    }

    // Write the computed value to output matrix C
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// PyTorch interface wrapper
torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    // Dimensions:
    // A: (K x M), B: (N x K) => C: (M x N)
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    const int TILE_SIZE = 16;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_transpose_shared_pad_kernel", ([&] {
        matmul_transpose_shared_pad_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Matrix multiplication with transposed inputs using shared memory with padding (CUDA)");
}
