#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE_M 32
#define BLOCK_SIZE_N 16
#define BLOCK_SIZE_K 16
#define UNROLL_FACTOR 4

// Kernel: each thread computes a 2x1 sub-tile (2 rows, 1 column) of C
// A is (K x M): element A[k, m] = A[k * M + m]
// B is (N x K): element B[n, k] = B[n * K + k]
// C is (M x N): element C[m, n] = C[m * N + n]

template <typename scalar_t>
__global__ void optimized_matmul_transpose_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {

    const int m_start = blockIdx.y * BLOCK_SIZE_M;
    const int n_start = blockIdx.x * BLOCK_SIZE_N;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Each thread computes two rows
    const int row0 = m_start + tx;
    const int row1 = row0 + (BLOCK_SIZE_M / 2);
    const int col = n_start + ty;

    __shared__ scalar_t A_tile[BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ scalar_t B_tile[BLOCK_SIZE_N][BLOCK_SIZE_K];

    // Register cache for accumulation
    scalar_t acc0 = 0;
    scalar_t acc1 = 0;

    // Thread ID for cooperative loading
    const int tId = ty * blockDim.x + tx;
    const int blockSize = blockDim.x * blockDim.y;

    // Main loop over tiles
    for (int tile = 0; tile < (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K; tile++) {
        // Cooperative loading of A tile with manual unrolling
        for (int i = tId; i < (BLOCK_SIZE_K * BLOCK_SIZE_M); i += blockSize) {
            const int k = i / BLOCK_SIZE_M;
            const int m = i % BLOCK_SIZE_M;
            const int global_k = tile * BLOCK_SIZE_K + k;
            const int global_m = m_start + m;
            A_tile[k][m] = (global_k < K && global_m < M) ? __ldg(&A[global_k * M + global_m]) : 0;
        }

        // Cooperative loading of B tile with manual unrolling
        for (int i = tId; i < (BLOCK_SIZE_N * BLOCK_SIZE_K); i += blockSize) {
            const int n = i / BLOCK_SIZE_K;
            const int k = i % BLOCK_SIZE_K;
            const int global_k = tile * BLOCK_SIZE_K + k;
            const int global_n = n_start + n;
            B_tile[n][k] = (global_k < K && global_n < N) ? __ldg(&B[global_n * K + global_k]) : 0;
        }

        __syncthreads();

        // Manual unrolling of the reduction loop
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; k += UNROLL_FACTOR) {
            // Load A values for both rows
            scalar_t a0[UNROLL_FACTOR], a1[UNROLL_FACTOR];
            scalar_t b[UNROLL_FACTOR];
            
            #pragma unroll
            for (int u = 0; u < UNROLL_FACTOR; u++) {
                if (k + u < BLOCK_SIZE_K) {
                    a0[u] = A_tile[k + u][tx];
                    a1[u] = A_tile[k + u][tx + (BLOCK_SIZE_M / 2)];
                    b[u] = B_tile[ty][k + u];
                }
            }

            // Compute partial products
            #pragma unroll
            for (int u = 0; u < UNROLL_FACTOR; u++) {
                if (k + u < BLOCK_SIZE_K) {
                    acc0 = __fmaf_rn(a0[u], b[u], acc0);
                    acc1 = __fmaf_rn(a1[u], b[u], acc1);
                }
            }
        }

        __syncthreads();
    }

    // Write results
    if (row0 < M && col < N) {
        C[row0 * N + col] = acc0;
    }
    if (row1 < M && col < N) {
        C[row1 * N + col] = acc1;
    }
}

torch::Tensor optimized_matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(16, 16);
    dim3 blocks((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
                (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "optimized_matmul_transpose_kernel", ([&] {
        optimized_matmul_transpose_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_matmul_transpose_cuda, "Optimized matrix multiplication with transpose (CUDA)");
}