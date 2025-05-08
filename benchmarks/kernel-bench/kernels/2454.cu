#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile dimensions
#define BLOCK_SIZE_M 32  // Each block computes 32 rows of C
#define BLOCK_SIZE_N 16  // Each block computes 16 columns of C
#define BLOCK_SIZE_K 16  // Reduction tile depth

// Inline device function to perform asynchronous copy from global to shared memory using cp.async
// This intrinsic is supported on NVIDIA architectures (Ampere, Hopper, etc.)

__device__ inline void cp_async(void* dst, const void* src, size_t count) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                  :
                  : "r"(dst), "l"(src), "n"(count)
                  : "memory");
}

// Kernel implementing double-buffering with asynchronous memory copies to overlap
// global memory transfers with computation. It computes C = A.T * B.T,
// where A is of dimensions (K x M) and B is (N x K), yielding C of dimensions (M x N).
// The kernel uses two shared memory buffers for each tile from A and B.

template <typename scalar_t>
__global__ void matmul_transpose_async_pipeline_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int M,
    int N,
    int K) {

    // Allocate double-buffered shared memory in linear layout
    // A_stage: two buffers, each holding BLOCK_SIZE_K * BLOCK_SIZE_M elements
    // B_stage: two buffers, each holding BLOCK_SIZE_N * BLOCK_SIZE_K elements
    __shared__ scalar_t A_stage[2][BLOCK_SIZE_K * BLOCK_SIZE_M];
    __shared__ scalar_t B_stage[2][BLOCK_SIZE_N * BLOCK_SIZE_K];

    // Compute starting indices for the tile computed by this block
    int m_start = blockIdx.y * BLOCK_SIZE_M;  // starting row in C
    int n_start = blockIdx.x * BLOCK_SIZE_N;  // starting column in C

    // Thread indices within block
    // We use a 16x16 block so that each thread computes two rows of C
    int tx = threadIdx.x; // range [0, 15]
    int ty = threadIdx.y; // range [0, 15]

    // Each thread computes two output rows: one at row0 and one at row1
    int row0 = m_start + tx;
    int row1 = row0 + (BLOCK_SIZE_M / 2); // offset by 16
    int col = n_start + ty;

    // Accumulators for the two outputs
    scalar_t acc0 = 0;
    scalar_t acc1 = 0;

    // Compute linear thread ID within the block
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y; // expected 256

    // Total number of elements per tile load
    const int tileA_elems = BLOCK_SIZE_K * BLOCK_SIZE_M; // 16*32 = 512
    const int tileB_elems = BLOCK_SIZE_N * BLOCK_SIZE_K;   // 16*16 = 256

    // Number of tiles needed along K dimension
    int numTiles = (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;

    // Setup double-buffer indices
    int cur_buf = 0;
    int next_buf = 1;

    // Preload the first tile (tile 0) into the current buffer asynchronously
    for (int idx = tid; idx < tileA_elems; idx += blockSize) {
        int k = idx / BLOCK_SIZE_M;
        int m = idx % BLOCK_SIZE_M;
        int global_k = 0 * BLOCK_SIZE_K + k;
        int global_m = m_start + m;
        if (global_k < K && global_m < M)
            cp_async(&A_stage[cur_buf][idx], &A[global_k * M + global_m], sizeof(scalar_t));
        else
            A_stage[cur_buf][idx] = 0;
    }
    for (int idx = tid; idx < tileB_elems; idx += blockSize) {
        int n = idx / BLOCK_SIZE_K;
        int k = idx % BLOCK_SIZE_K;
        int global_n = n_start + n;
        int global_k = 0 * BLOCK_SIZE_K + k;
        if (global_n < N && global_k < K)
            cp_async(&B_stage[cur_buf][idx], &B[global_n * K + global_k], sizeof(scalar_t));
        else
            B_stage[cur_buf][idx] = 0;
    }
    // Commit the asynchronous copies and wait for them to complete
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    // Loop over all tiles
    for (int tile = 0; tile < numTiles; tile++) {
        // If a next tile exists, preload it into the next buffer
        if (tile < numTiles - 1) {
            int next_tile = tile + 1;
            for (int idx = tid; idx < tileA_elems; idx += blockSize) {
                int k = idx / BLOCK_SIZE_M;
                int m = idx % BLOCK_SIZE_M;
                int global_k = next_tile * BLOCK_SIZE_K + k;
                int global_m = m_start + m;
                if (global_k < K && global_m < M)
                    cp_async(&A_stage[next_buf][idx], &A[global_k * M + global_m], sizeof(scalar_t));
                else
                    A_stage[next_buf][idx] = 0;
            }
            for (int idx = tid; idx < tileB_elems; idx += blockSize) {
                int n = idx / BLOCK_SIZE_K;
                int k = idx % BLOCK_SIZE_K;
                int global_n = n_start + n;
                int global_k = next_tile * BLOCK_SIZE_K + k;
                if (global_n < N && global_k < K)
                    cp_async(&B_stage[next_buf][idx], &B[global_n * K + global_k], sizeof(scalar_t));
                else
                    B_stage[next_buf][idx] = 0;
            }
            asm volatile("cp.async.commit_group;\n");
        }

        // Ensure the current tile's data is ready before computation
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();

        // Compute partial results using the currently loaded tile in the current buffer
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; k++) {
            scalar_t a0 = A_stage[cur_buf][k * BLOCK_SIZE_M + tx];
            scalar_t a1 = A_stage[cur_buf][k * BLOCK_SIZE_M + tx + (BLOCK_SIZE_M / 2)];
            scalar_t b_val = B_stage[cur_buf][ty * BLOCK_SIZE_K + k];
            acc0 += a0 * b_val;
            acc1 += a1 * b_val;
        }
        __syncthreads();

        // Swap the buffers for double buffering
        int temp = cur_buf;
        cur_buf = next_buf;
        next_buf = temp;
    }

    // Write the computed outputs to global memory
    if (row0 < M && col < N)
        C[row0 * N + col] = acc0;
    if (row1 < M && col < N)
        C[row1 * N + col] = acc1;
}

// PyTorch binding: launches the asynchronous pipelined kernel

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    // A: (K x M), B: (N x K), so C: (M x N)
    int K = A.size(0);
    int M = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    // Launch parameters: using a 16x16 thread block corresponds to a 32x16 tile of C
    dim3 threads(16, 16);
    dim3 blocks((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_transpose_async_pipeline_kernel", ([&] {
        matmul_transpose_async_pipeline_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K);
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Matmul with transposed inputs using async pipeline (CUDA)");
}
