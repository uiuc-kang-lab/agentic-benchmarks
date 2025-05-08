#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile size
#define TILE_SIZE 16

// Modular device function to load a tile from matrix A into shared memory
// A is stored as (K x M): A[k * M + m], where m is the row index
template <typename scalar_t>
__device__ inline void load_A_tile(const scalar_t* __restrict__ A,
                                    scalar_t A_tile[TILE_SIZE][TILE_SIZE],
                                    int tile_idx, int row, int M, int K,
                                    int tx, int ty) {
    int global_k = tile_idx * TILE_SIZE + ty; // k index for the tile
    if (row < M && global_k < K)
        A_tile[ty][tx] = A[global_k * M + row];
    else
        A_tile[ty][tx] = static_cast<scalar_t>(0);
}

// Modular device function to load a tile from matrix B into shared memory
// B is stored as (N x K): B[n * K + k], where n is the column index in output C
template <typename scalar_t>
__device__ inline void load_B_tile(const scalar_t* __restrict__ B,
                                    scalar_t B_tile[TILE_SIZE][TILE_SIZE],
                                    int tile_idx, int col, int N, int K,
                                    int tx, int ty) {
    int global_k = tile_idx * TILE_SIZE + tx; // k index for the tile
    if (col < N && global_k < K)
        B_tile[ty][tx] = B[col * K + global_k];
    else
        B_tile[ty][tx] = static_cast<scalar_t>(0);
}

// Modular device function to compute dot product from the loaded tiles
// Each thread computes its partial sum for one output element
template <typename scalar_t>
__device__ inline scalar_t compute_tile_dot(const scalar_t A_tile[TILE_SIZE][TILE_SIZE],
                                               const scalar_t B_tile[TILE_SIZE][TILE_SIZE],
                                               int tx, int ty, int tile_bound) {
    scalar_t sum = 0;
    for (int k = 0; k < tile_bound; k++) {
        sum += A_tile[k][tx] * B_tile[ty][k];
    }
    return sum;
}

// Kernel: Each thread computes one element of the output matrix C
// C[m, n] = sum_{k} A[k, m] * B[n, k]
// where A and B are stored in transposed forms compared to standard layout
template <typename scalar_t>
__global__ void matmul_transpose_modular_kernel(
    const scalar_t* __restrict__ A,   // A: (K x M)
    const scalar_t* __restrict__ B,   // B: (N x K)
    scalar_t* __restrict__ C,         // C: (M x N)
    int M, int N, int K) {

    // Compute global row and column for C
    int row = blockIdx.x * TILE_SIZE + threadIdx.x; // corresponds to m
    int col = blockIdx.y * TILE_SIZE + threadIdx.y; // corresponds to n

    scalar_t acc = 0;

    // Allocate shared memory for tiles
    __shared__ scalar_t A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t B_tile[TILE_SIZE][TILE_SIZE];

    // Loop over tiles along k dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        // Load tile from A and B using modular device functions
        load_A_tile<scalar_t>(A, A_tile, t, row, M, K, threadIdx.x, threadIdx.y);
        load_B_tile<scalar_t>(B, B_tile, t, col, N, K, threadIdx.x, threadIdx.y);

        __syncthreads();

        // Determine actual number of iterations for this tile (handles boundary cases)
        int tile_bound = TILE_SIZE;
        if (t == numTiles - 1) {
            int remainder = K - t * TILE_SIZE;
            tile_bound = remainder < TILE_SIZE ? remainder : TILE_SIZE;
        }

        // Compute partial dot product from the current tile
        acc += compute_tile_dot<scalar_t>(A_tile, B_tile, threadIdx.x, threadIdx.y, tile_bound);

        __syncthreads();
    }

    // Write the result to the output matrix C if within valid bounds
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// PyTorch binding

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    // Dimensions:
    // A: (K x M), B: (N x K) => C: (M x N)
    int K = A.size(0);
    int M = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((M + TILE_SIZE - 1) / TILE_SIZE,
                (N + TILE_SIZE - 1) / TILE_SIZE);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_transpose_modular_kernel", ([&] {
        matmul_transpose_modular_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K);
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Matrix multiplication with transposed inputs using modular device functions (CUDA)");
}
