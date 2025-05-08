#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses shared memory tiling with branchless loads
// employing arithmetic masks and index clamping to avoid divergent branches within warps.
// It computes C = A.T * B.T, where A is of shape (K, M) and B is of shape (N, K).

template <typename scalar_t>
__global__ void matmul_transpose_shared_kernel_nodiv(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M, // A: (K, M), output has M rows
    const int N, // B: (N, K), output has N columns
    const int K  // inner dimension
) {
    const int TILE_SIZE = 32;
    __shared__ scalar_t A_shared[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t B_shared[TILE_SIZE][TILE_SIZE];

    // Compute global row and column indices for C
    int row = blockIdx.x * TILE_SIZE + threadIdx.x;
    int col = blockIdx.y * TILE_SIZE + threadIdx.y;
    scalar_t sum = 0;

    // Number of tiles along the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < numTiles; tile++) {
        // Compute the index in the K dimension for A and B
        int a_k = tile * TILE_SIZE + threadIdx.y;  // for A: index in K dimension
        int b_k = tile * TILE_SIZE + threadIdx.x;  // for B: index in K dimension

        // Compute branchless masks and clamp indices to avoid out-of-bound accesses
        // For A: A is stored as (K, M) so element index = (tile*TILE_SIZE+threadIdx.y) * M + row
        int validA = (a_k < K && row < M) ? 1 : 0;
        int clamped_a_k = (a_k < K) ? a_k : (K - 1);
        // Use a safe row index for out-of-bound threads
        int safe_row = (row < M) ? row : 0;
        A_shared[threadIdx.y][threadIdx.x] = A[clamped_a_k * M + safe_row] * static_cast<scalar_t>(validA);

        // For B: B is stored as (N, K) so element index = col * K + (tile*TILE_SIZE+threadIdx.x)
        int validB = (b_k < K && col < N) ? 1 : 0;
        int clamped_b_k = (b_k < K) ? b_k : (K - 1);
        int safe_col = (col < N) ? col : 0;
        B_shared[threadIdx.x][threadIdx.y] = B[safe_col * K + clamped_b_k] * static_cast<scalar_t>(validB);

        __syncthreads();

        // Perform the dot-product for the current tile.
        // The loop is fully unrolled and uniform across warps.
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += A_shared[k][threadIdx.x] * B_shared[k][threadIdx.y];
        }
        __syncthreads();
    }

    // Write the computed sum to C if within valid output bounds
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host function to launch the CUDA kernel
torch::Tensor matmul_transpose_cuda_nodiv(torch::Tensor A, torch::Tensor B) {
    // Dimensions: A is (K, M) and B is (N, K), so C will be (M, N)
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());
    
    const int TILE_SIZE = 32;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((M + TILE_SIZE - 1) / TILE_SIZE,
                (N + TILE_SIZE - 1) / TILE_SIZE);

    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_shared_kernel_nodiv", ([&] {
        matmul_transpose_shared_kernel_nodiv<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda_nodiv, "Matrix multiplication with transposed inputs using shared memory and branchless loads (CUDA)");
}
