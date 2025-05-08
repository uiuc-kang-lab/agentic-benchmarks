#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel computes C = A.T * B.T using shared memory tiling while minimizing
// warp divergence by replacing conditional loads with branchless arithmetic.
// It uses clamping with min() and multiplies by valid masks to avoid divergent control flow.
// A is of shape (K, M) and B is of shape (N, K), producing C of shape (M, N).

template <typename scalar_t>
__global__ void matmul_transpose_branchless_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,  // Number of rows in output (A's second dim)
    const int N,  // Number of columns in output (B's first dim)
    const int K   // Shared dimension (A's first dim, B's second dim)
) {
    const int TILE_SIZE = 32;
    __shared__ scalar_t A_shared[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts
    __shared__ scalar_t B_shared[TILE_SIZE][TILE_SIZE + 1];

    // Compute global indices for C
    int row = blockIdx.x * TILE_SIZE + threadIdx.x;
    int col = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    scalar_t sum = 0;
    
    // Loop over tiles along the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile = 0; tile < numTiles; tile++) {
        // Calculate global index for loading from A and B
        int a_k = tile * TILE_SIZE + ty; // for A (stored as (K, M))
        int b_k = tile * TILE_SIZE + tx; // for B (stored as (N, K))

        // Use branchless logic to load A: clamp indices and apply a valid mask
        int validA = (a_k < K && row < M) ? 1 : 0;
        int a_index = (a_k < K) ? a_k : (K - 1);
        int safe_row = (row < M) ? row : 0;
        A_shared[ty][tx] = A[a_index * M + safe_row] * static_cast<scalar_t>(validA);

        // Similarly load B with branchless conditional logic
        int validB = (b_k < K && col < N) ? 1 : 0;
        int b_index = (b_k < K) ? b_k : (K - 1);
        int safe_col = (col < N) ? col : 0;
        B_shared[tx][ty] = B[safe_col * K + b_index] * static_cast<scalar_t>(validB);

        __syncthreads();

        // Multiply the two tiles together and accumulate the results.
        // Loop is unrolled for performance and has uniform control flow across warps.
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += A_shared[i][tx] * B_shared[i][ty];
        }
        
        __syncthreads();
    }
    
    // Write the computed sum to C using boundary check
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host function to launch the branchless CUDA kernel
torch::Tensor matmul_transpose_cuda_branchless(torch::Tensor A, torch::Tensor B) {
    // Dimensions: A is (K, M) and B is (N, K), so C is (M, N)
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);
    
    auto C = torch::empty({M, N}, A.options());
    
    const int TILE_SIZE = 32;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((M + TILE_SIZE - 1) / TILE_SIZE,
                (N + TILE_SIZE - 1) / TILE_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_branchless_kernel", ([&] {
        matmul_transpose_branchless_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda_branchless, "Matrix multiplication with transposed matrices using branchless loads to avoid warp divergence (CUDA)");
}
