#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile sizes
#define BLOCK_SIZE_M 32  // Output tile height (each block computes 32 rows)
#define BLOCK_SIZE_N 16  // Output tile width (each block computes 16 columns)
#define BLOCK_SIZE_K 16  // Reduction tile depth

// Hybrid Kernel: Combines tiling and warp-level primitives
// A is (K x M): element A[k, m] = A[k * M + m]
// B is (N x K): element B[n, k] = B[n * K + k]
// C is (M x N): element C[m, n] = C[m * N + n]

template <typename scalar_t>
__global__ void matmul_transpose_hybrid_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int M,
    int N,
    int K) {

    // Determine the starting indices for this block's tile in C
    int m_start = blockIdx.y * BLOCK_SIZE_M;  // row start in C
    int n_start = blockIdx.x * BLOCK_SIZE_N;  // col start in C

    // Thread indices within the block
    int tx = threadIdx.x; // Expected range: [0, 15]
    int ty = threadIdx.y; // Expected range: [0, 15]

    // Each thread computes two rows: row0 and row1
    int row0 = m_start + tx;             // first row computed by this thread
    int row1 = row0 + (BLOCK_SIZE_M / 2);  // second row computed (offset by 16)
    int col = n_start + ty;              // column index in C

    // Accumulators for the two output elements
    scalar_t acc0 = 0;
    scalar_t acc1 = 0;

    // Declare shared memory tiles
    __shared__ scalar_t A_tile[BLOCK_SIZE_K][BLOCK_SIZE_M]; // Size: 16 x 32
    __shared__ scalar_t B_tile[BLOCK_SIZE_N][BLOCK_SIZE_K];   // Size: 16 x 16

    int numTiles = (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;
    for (int tile = 0; tile < numTiles; tile++) {
        // Load A tile into shared memory
        int kd = threadIdx.x;
        int md = threadIdx.y;
        int global_m = m_start + md;
        int global_k = tile * BLOCK_SIZE_K + kd;
        if (global_m < M && global_k < K)
            A_tile[kd][md] = A[global_k * M + global_m];
        else
            A_tile[kd][md] = 0;

        // Load B tile into shared memory
        int nd = threadIdx.x;
        kd = threadIdx.y;
        int global_n = n_start + nd;
        global_k = tile * BLOCK_SIZE_K + kd;
        if (global_n < N && global_k < K)
            B_tile[nd][kd] = B[global_n * K + global_k];
        else
            B_tile[nd][kd] = 0;

        __syncthreads();

        // Compute the partial results for this tile
        for (int k = 0; k < BLOCK_SIZE_K; k++) {
            scalar_t a_val0 = A_tile[k][tx];                     // for row0
            scalar_t a_val1 = A_tile[k][tx + (BLOCK_SIZE_M / 2)];  // for row1
            scalar_t b_val = B_tile[ty][k];
            acc0 += a_val0 * b_val;
            acc1 += a_val1 * b_val;
        }
        __syncthreads();
    }

    // Perform warp-level reduction using __shfl_down_sync to sum partial results
    for (int offset = 16; offset > 0; offset /= 2) {
        acc0 += __shfl_down_sync(0xffffffff, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffff, acc1, offset);
    }

    // Write the results to global memory
    if (tx == 0) {
        if (row0 < M && col < N) {
            C[row0 * N + col] = acc0;
        }
        if (row1 < M && col < N) {
            C[row1 * N + col] = acc1;
        }
    }
}

// PyTorch binding

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    // Dimensions:
    // A: (K x M), B: (N x K), therefore C: (M x N)
    int K = A.size(0);
    int M = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    // Define block dimensions: use 16x16 threads per block
    dim3 threads(16, 16);
    // Grid dimensions based on tile sizes
    dim3 blocks((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_transpose_hybrid_kernel", ([&] {
        matmul_transpose_hybrid_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K);
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Matrix multiplication with transposed inputs using hybrid kernel (CUDA)");
}
