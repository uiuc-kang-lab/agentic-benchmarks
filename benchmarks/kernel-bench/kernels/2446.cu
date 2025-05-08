#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile configuration: Each block computes a TILE_ROWS x TILE_COLS tile of C.
// Each block uses 32*(TILE_ROWS*TILE_COLS) threads (i.e. one warp per output element).
// We choose a larger TILE_K to allow loop unrolling in the inner reduction.

#define TILE_ROWS 4      // Number of output rows per block tile
#define TILE_COLS 2      // Number of output columns per block tile
#define TILE_K 128       // Tile size in the reduction (k) dimension

// The kernel uses warp-level reduction combined with manual loop unrolling to improve ILP

template <typename scalar_t>
__global__ void hybrid_unrolled_warp_kernel(
    const scalar_t* __restrict__ A,  // A: (K x M) stored as A[k * M + m]
    const scalar_t* __restrict__ B,  // B: (N x K) stored as B[n * K + k]
    scalar_t* __restrict__ C,        // C: (M x N) stored as C[m * N + n]
    const int M,                     // Number of rows in C (and A’s 2nd dim)
    const int N,                     // Number of columns in C (and B’s 1st dim)
    const int K) {                   // Reduction dimension (and A’s 1st, B’s 2nd)

    // Compute starting indices of the tile in output matrix C
    int m_start = blockIdx.y * TILE_ROWS;
    int n_start = blockIdx.x * TILE_COLS;

    // Each block uses 32*(TILE_ROWS*TILE_COLS) threads (e.g., 32*8=256 threads).
    // Each warp (32 threads) is assigned to compute one output element of C in the tile
    int warp_id = threadIdx.x / 32;  // Warp index within the block tile
    int lane = threadIdx.x & 31;     // Lane index within the warp

    // Map warp_id to a specific output element in the tile
    int local_row = warp_id % TILE_ROWS;  // row index within the tile
    int local_col = warp_id / TILE_ROWS;  // col index within the tile

    int global_row = m_start + local_row;
    int global_col = n_start + local_col;

    // Accumulator for the dot product
    scalar_t acc = 0;

    // Allocate shared memory for the current k-tile for A and B
    // For A: load TILE_ROWS rows, for B: load TILE_COLS columns
    __shared__ scalar_t As[TILE_ROWS][TILE_K];
    __shared__ scalar_t Bs[TILE_COLS][TILE_K];

    // Loop over tiles along the k-dimension
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Cooperative loading of A tile into shared memory
        int total_A = TILE_ROWS * TILE_K;
        for (int idx = threadIdx.x; idx < total_A; idx += blockDim.x) {
            int row = idx / TILE_K;       // local row index in the A tile
            int t = idx % TILE_K;           // local k index within the tile
            int global_k = k_tile + t;
            int global_m = m_start + row;
            if (global_m < M && global_k < K) {
                // A is stored in row-major order per k slice: A[global_k * M + global_m]
                As[row][t] = A[global_k * M + global_m];
            } else {
                As[row][t] = 0;
            }
        }

        // Cooperative loading of B tile into shared memory
        int total_B = TILE_COLS * TILE_K;
        for (int idx = threadIdx.x; idx < total_B; idx += blockDim.x) {
            int col = idx / TILE_K;      // local col index in the B tile
            int t = idx % TILE_K;          // local k index
            int global_k = k_tile + t;
            int global_n = n_start + col;
            if (global_n < N && global_k < K) {
                // B is stored as B[global_n * K + global_k]
                Bs[col][t] = B[global_n * K + global_k];
            } else {
                Bs[col][t] = 0;
            }
        }

        __syncthreads();  // Ensure tiles are loaded

        // Each warp computes the dot product for its assigned output element:
        // Dot-product: sum_{t=0}^{TILE_K-1} As[local_row][t] * Bs[local_col][t]
        scalar_t sum_tile = 0;
        // Distribute the work within the warp using a strided loop
        // Since TILE_K is large (128), each lane performs several iterations
        #pragma unroll
        for (int t = lane; t < TILE_K; t += 32) {
            // Use fused multiply-add for precision and performance
            sum_tile = __fmaf_rn(As[local_row][t], Bs[local_col][t], sum_tile);
        }

        // Warp-level reduction using shuffle intrinsics
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_tile += __shfl_down_sync(0xffffffff, sum_tile, offset);
        }

        // Only lane 0 of each warp accumulate the result from this tile
        if (lane == 0) {
            acc += sum_tile;
        }

        __syncthreads(); // Ensure shared memory tiles can be overwritten in next iteration
    }

    // Write the final result for the output element computed by the warp
    if (global_row < M && global_col < N && lane == 0) {
        C[global_row * N + global_col] = acc;
    }
}

// CUDA interface function for PyTorch binding

torch::Tensor hybrid_unrolled_warp_cuda(torch::Tensor A, torch::Tensor B) {
    // Matrix dimensions:
    // A: (K x M),    B: (N x K)  =>  C: (M x N)
    int K = A.size(0);
    int M = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    // Grid: Each block computes a tile of size (TILE_ROWS x TILE_COLS)
    dim3 blocks((N + TILE_COLS - 1) / TILE_COLS, (M + TILE_ROWS - 1) / TILE_ROWS);
    // Each block has 32 * (TILE_ROWS * TILE_COLS) threads, e.g., 32 * 8 = 256
    int threads_per_block = 32 * (TILE_ROWS * TILE_COLS);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "hybrid_unrolled_warp_kernel", ([&] {
        hybrid_unrolled_warp_kernel<scalar_t><<<blocks, threads_per_block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &hybrid_unrolled_warp_cuda, "Efficient matrix multiplication with tiling, unrolling and warp-level reduction (CUDA)");
}
