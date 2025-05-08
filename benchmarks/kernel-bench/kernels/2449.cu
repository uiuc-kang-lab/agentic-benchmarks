#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tunable tile sizes for exploring optimal block configurations.
// These can be modified to experiment with different block sizes (e.g., 32, 64, 128, 256, 512 threads per block).
#ifndef TILE_M
#define TILE_M 32  // Tile height in output C. Each block computes TILE_M rows, with each thread computing 2 rows.
#endif

#ifndef TILE_N
#define TILE_N 16  // Tile width in output C. Each block computes TILE_N columns.
#endif

#ifndef TILE_K
#define TILE_K 16  // Tile depth for the reduction (K dimension).
#endif

// Kernel: Each thread computes 2 output elements (covering 2 rows and 1 column) of C.
// A is transposed: A is stored with dimensions (K x M) and accessed as A[k * M + m].
// B is transposed: B is stored with dimensions (N x K) and accessed as B[n * K + k].
// C is stored with dimensions (M x N) and accessed as C[m * N + n].

// The thread block is configured as (TILE_M/2, TILE_N), so total threads per block = (TILE_M/2) * TILE_N.
// For example, with TILE_M=32 and TILE_N=16, we have 16*16 = 256 threads per block.

template <typename scalar_t>
__global__ void matmul_transpose_tunable_kernel(
    const scalar_t* __restrict__ A, // A: (K x M)
    const scalar_t* __restrict__ B, // B: (N x K)
    scalar_t* __restrict__ C,       // C: (M x N)
    int M, int N, int K) {

    // Compute the starting row and column for this block's tile in C.
    int m_start = blockIdx.y * TILE_M;  // starting row in C
    int n_start = blockIdx.x * TILE_N;  // starting column in C

    // Thread block dimensions: threadIdx.x in [0, TILE_M/2) and threadIdx.y in [0, TILE_N)
    int tx = threadIdx.x; // used for row indexing; each thread computes 2 rows
    int ty = threadIdx.y; // used for column indexing

    // Each thread computes two rows:
    int row0 = m_start + tx;
    int row1 = m_start + tx + TILE_M/2;
    int col  = n_start + ty;

    // Accumulators for the two output elements computed by each thread.
    scalar_t acc0 = 0;
    scalar_t acc1 = 0;

    // Shared memory tiles for A and B.
    __shared__ scalar_t A_tile[TILE_K][TILE_M]; // Holds a tile from A (dimensions: TILE_K x TILE_M)
    __shared__ scalar_t B_tile[TILE_N][TILE_K];   // Holds a tile from B (dimensions: TILE_N x TILE_K)

    // Total threads per block
    int tId = threadIdx.y * (TILE_M/2) + threadIdx.x;
    int blockSize = (TILE_M/2) * TILE_N;

    // Number of tiles needed to cover the K dimension.
    int numTiles = (K + TILE_K - 1) / TILE_K;
    
    for (int tile = 0; tile < numTiles; tile++) {
        // Cooperative loading of A tile into shared memory.
        int totalA = TILE_K * TILE_M; // Total elements in A_tile
        for (int idx = tId; idx < totalA; idx += blockSize) {
            int k_idx = idx / TILE_M; // row index within the tile (in K dimension)
            int m_idx = idx % TILE_M; // column index within the tile (in M dimension)
            int global_k = tile * TILE_K + k_idx;
            int global_m = m_start + m_idx;
            if (global_k < K && global_m < M)
                A_tile[k_idx][m_idx] = __ldg(&A[global_k * M + global_m]);
            else
                A_tile[k_idx][m_idx] = 0;
        }

        // Cooperative loading of B tile into shared memory.
        int totalB = TILE_N * TILE_K; // Total elements in B_tile
        for (int idx = tId; idx < totalB; idx += blockSize) {
            int n_idx = idx / TILE_K; // row index within the tile (in N dimension)
            int k_idx = idx % TILE_K; // column index within the tile (in K dimension)
            int global_n = n_start + n_idx;
            int global_k = tile * TILE_K + k_idx;
            if (global_n < N && global_k < K)
                B_tile[n_idx][k_idx] = __ldg(&B[global_n * K + global_k]);
            else
                B_tile[n_idx][k_idx] = 0;
        }
        __syncthreads();

        // Compute partial results for the current tile.
        for (int k = 0; k < TILE_K; k++) {
            scalar_t a0 = A_tile[k][tx];                    // value for the first row
            scalar_t a1 = A_tile[k][tx + TILE_M/2];         // value for the second row
            scalar_t b  = B_tile[ty][k];
            acc0 += a0 * b;
            acc1 += a1 * b;
        }
        __syncthreads();
    }

    // Write the computed values to global memory if within bounds.
    if (row0 < M && col < N)
        C[row0 * N + col] = acc0;
    if (row1 < M && col < N)
        C[row1 * N + col] = acc1;
}


// PyTorch binding
// A: (K x M), B: (N x K) ==> C: (M x N)

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    int K = A.size(0);
    int M = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    // Configure thread block dimensions based on tunable tile sizes.
    // Threads per block: (TILE_M/2, TILE_N). For default values, (32/2, 16) = (16,16) = 256 threads per block.
    dim3 threads(TILE_M/2, TILE_N);
    // Grid dimensions: each block covers a TILE_M x TILE_N tile of the output matrix C.
    dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_transpose_tunable_kernel", ([&] {
        matmul_transpose_tunable_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K);
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Tunable block size matrix multiplication with transposed inputs (CUDA)");
}
