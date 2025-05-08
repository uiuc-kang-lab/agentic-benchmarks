#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile sizes
#define BLOCK_SIZE_M 32  // Output tile height (each block computes 32 rows)
#define BLOCK_SIZE_N 16  // Output tile width (each block computes 16 columns)
#define BLOCK_SIZE_K 16  // Reduction tile depth

// Kernel: each thread computes a 2x1 sub-tile (2 rows, 1 column) of C
// A is (K x M): element A[k, m] = A[k * M + m]
// B is (N x K): element B[n, k] = B[n * K + k]
// C is (M x N): element C[m, n] = C[m * N + n]

__global__ void matmul_transpose_unroll_base_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
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
    float acc0 = 0;
    float acc1 = 0;

    // Declare shared memory tiles
    __shared__ float A_tile[BLOCK_SIZE_K][BLOCK_SIZE_M]; // Size: 16 x 32
    __shared__ float B_tile[BLOCK_SIZE_N][BLOCK_SIZE_K]; // Size: 16 x 16

    int numTiles = (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;
    for (int tile = 0; tile < numTiles; tile++) {
        // Load A tile into shared memory
        if (row0 < M && (tile * BLOCK_SIZE_K + ty) < K) {
            A_tile[ty][tx] = A[(tile * BLOCK_SIZE_K + ty) * M + row0];
            A_tile[ty][tx + BLOCK_SIZE_M / 2] = A[(tile * BLOCK_SIZE_K + ty) * M + row1];
        }
        
        // Load B tile into shared memory
        if (col < N && (tile * BLOCK_SIZE_K + tx) < K) {
            B_tile[tx][ty] = B[col * K + (tile * BLOCK_SIZE_K + tx)];
        }

        __syncthreads();

        // Compute the partial results for this tile
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; k++) {
            float a_val0 = A_tile[k][tx];
            float a_val1 = A_tile[k][tx + (BLOCK_SIZE_M / 2)];
            float b_val = B_tile[ty][k];
            acc0 += a_val0 * b_val;
            acc1 += a_val1 * b_val;
        }

        __syncthreads();
    }

    // Write the results to global memory
    if (row0 < M && col < N) {
        C[row0 * N + col] = acc0;
    }
    if (row1 < M && col < N) {
        C[row1 * N + col] = acc1;
    }
}


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

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_transpose_unroll_base_kernel", ([&] {
        matmul_transpose_unroll_base_kernel<<<blocks, threads>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, N, K);
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Matrix multiplication with transposed inputs using unrolled kernel (CUDA)");
}
