/*
Hybrid Tiled Warp Kernel:
This CUDA kernel combines the shared-memory tiling strategy (as in Kernel 1) with warp-level reduction (as in Kernel 2).
Each block computes a small tile of the output matrix C, and each warp within the block collaboratively computes one output element.
The kernel cooperatively loads a tile of A and B into shared memory, then each warp iterates over the k-dimension in chunks, using its 32 threads to compute partial dot products, which are then reduced using warp shuffles.
This approach improves global memory reuse and reduces synchronization overhead in the reduction step.

Matrix dimensions (as in the original kernels):
 A: (K x M) stored such that A[k, m] = A[k * M + m]
 B: (N x K) stored such that B[n, k] = B[n * K + k]
 C: (M x N) stored such that C[m, n] = C[m * N + n]

Tile parameters for this kernel are chosen as follows:
  tile_rows = 4,  tile_cols = 2, and tile_k = 16.
Each block will use (tile_rows * tile_cols) warps, i.e., blockDim.x = 32 * (4*2) = 256 threads.
Each warp is assigned one output element in the block tile, with its global indices computed based on block indices and warp index.

Below is the full code including the PyTorch binding.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile parameters
#define TILE_ROWS 4    // number of output rows per block tile
#define TILE_COLS 2    // number of output columns per block tile
#define TILE_K 16      // tile size in the reduction (k) dimension

// Each block uses (TILE_ROWS * TILE_COLS) warps, i.e., blockDim.x = 32 * (TILE_ROWS * TILE_COLS) = 256 threads

template <typename scalar_t>
__global__ void hybrid_tiled_warp_kernel(
    const scalar_t* __restrict__ A,  // A: (K x M) matrix
    const scalar_t* __restrict__ B,  // B: (N x K) matrix
    scalar_t* __restrict__ C,        // C: (M x N) matrix
    const int M,                     // number of rows in C (and A's second dim)
    const int N,                     // number of cols in C (and B's first dim)
    const int K) {                   // reduction dimension (and A's first dim, B's second dim)

    // Determine the output tile starting indices for this block
    int m_start = blockIdx.y * TILE_ROWS;  // starting row index for C
    int n_start = blockIdx.x * TILE_COLS;    // starting col index for C

    // Each warp in the block computes one output element in the tile
    // Expect blockDim.x == 32 * (TILE_ROWS * TILE_COLS)
    int warp_id = threadIdx.x / 32;          // warp id within the block
    int lane = threadIdx.x & 31;             // lane id within the warp

    // Map warp_id to a specific output within the block tile:
    // For example, with TILE_ROWS=4 and TILE_COLS=2, there are 8 warps per block tile.
    // We assign: local_row = warp_id % TILE_ROWS, local_col = warp_id / TILE_ROWS
    int local_row = warp_id % TILE_ROWS;
    int local_col = warp_id / TILE_ROWS;

    int global_row = m_start + local_row;
    int global_col = n_start + local_col;

    // Accumulator for the dot-product
    scalar_t acc = 0;

    // Allocate shared memory for the current k-tile for A and B.
    // For A: we need data for each row in the tile (TILE_ROWS) for the k-tile
    // For B: we need data for each column in the tile (TILE_COLS) for the k-tile
    __shared__ scalar_t As[TILE_ROWS][TILE_K];  // shared tile from A
    __shared__ scalar_t Bs[TILE_COLS][TILE_K];    // shared tile from B

    // Loop over the k-dimension in chunks of TILE_K
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Cooperative loading of A tile into shared memory
        // Total number of elements for A tile: TILE_ROWS * TILE_K
        int total_A = TILE_ROWS * TILE_K;
        for (int idx = threadIdx.x; idx < total_A; idx += blockDim.x) {
            int r = idx / TILE_K;   // local row in tile
            int t = idx % TILE_K;   // position within the k-tile
            int global_k = k_tile + t;
            // Load A if within bounds: A is (K x M), access as A[global_k * M + (m_start + r)]
            if ((m_start + r) < M && global_k < K) {
                As[r][t] = A[global_k * M + (m_start + r)];
            } else {
                As[r][t] = 0;
            }
        }

        // Cooperative loading of B tile into shared memory
        // Total number of elements for B tile: TILE_COLS * TILE_K
        int total_B = TILE_COLS * TILE_K;
        for (int idx = threadIdx.x; idx < total_B; idx += blockDim.x) {
            int c = idx / TILE_K;   // local col in tile
            int t = idx % TILE_K;   // position within the k-tile
            int global_k = k_tile + t;
            // Load B if within bounds: B is (N x K), access as B[(n_start + c) * K + global_k]
            if ((n_start + c) < N && global_k < K) {
                Bs[c][t] = B[(n_start + c) * K + global_k];
            } else {
                Bs[c][t] = 0;
            }
        }

        __syncthreads();

        // Each warp computes partial dot product for its assigned output element:
        // The dot product is over the current k-tile: sum_{t=0}^{TILE_K} As[local_row][t] * Bs[local_col][t]
        // Distribute the work within the warp: each lane processes a subset of the TILE_K elements
        scalar_t sum_tile = 0;
        for (int t = lane; t < TILE_K; t += 32) {
            sum_tile += As[local_row][t] * Bs[local_col][t];
        }
        
        // Warp-level reduction to sum the partial products
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_tile += __shfl_down_sync(0xffffffff, sum_tile, offset);
        }

        // Only lane 0 of each warp adds the result of this tile to the accumulator
        if (lane == 0) {
            acc += sum_tile;
        }

        __syncthreads(); // Ensure all threads have finished with the shared memory tile
    }

    // Write the computed output element to global memory (only lane 0 of the warp does the write)
    if (global_row < M && global_col < N) {
        if (lane == 0) {
            C[global_row * N + global_col] = acc;
        }
    }
}


// CUDA interface function called from PyTorch
torch::Tensor hybrid_tiled_warp_cuda(torch::Tensor A, torch::Tensor B) {
    // Dimensions:
    // A: (K x M), B: (N x K), so that C: (M x N)
    int K = A.size(0);
    int M = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    // Define grid dimensions based on the tile sizes for output C
    // Each block computes a tile of size (TILE_ROWS x TILE_COLS), i.e., 4 x 2 outputs
    dim3 blocks((N + TILE_COLS - 1) / TILE_COLS, (M + TILE_ROWS - 1) / TILE_ROWS);
    // Each block has 32*(TILE_ROWS*TILE_COLS) threads
    int threads_per_block = 32 * (TILE_ROWS * TILE_COLS); // 32*8 = 256

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "hybrid_tiled_warp_kernel", ([&] {
        hybrid_tiled_warp_kernel<scalar_t><<<blocks, threads_per_block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &hybrid_tiled_warp_cuda, "Matrix multiplication with transposed inputs using hybrid tiled warp kernel (CUDA)");
}
