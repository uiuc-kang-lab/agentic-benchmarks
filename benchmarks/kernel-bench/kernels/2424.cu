#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that partitions the K dimension among blocks and uses atomicAdd to accumulate partial results.
// A is (K x M) and B is (N x K), so C is (M x N) where
// C[m, n] = sum_{k=0}^{K-1} A[k * M + m] * B[n * K + k]

template <typename scalar_t>
__global__ void matmul_transpose_atomic_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K,
    const int TILE_K) {

    // Determine the output element (m, n) handled by this thread
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    // Each block in the z-dimension processes a tile of the K dimension
    int k_start = blockIdx.z * TILE_K;
    int k_end = (k_start + TILE_K < K) ? (k_start + TILE_K) : K;

    if (m < M && n < N) {
        scalar_t partial_sum = 0;
        // Compute partial dot product for the assigned k-range
        for (int k = k_start; k < k_end; k++) {
            // A is stored as (K x M): A[k, m] = A[k * M + m]
            // B is stored as (N x K): B[n, k] = B[n * K + k]
            partial_sum += A[k * M + m] * B[n * K + k];
        }
        // Use atomicAdd to safely accumulate partial sums from different k-tiles
        atomicAdd(&C[m * N + n], partial_sum);
    }
}

// PyTorch CUDA interface
torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    // Dimensions: A is (K x M), B is (N x K), so C is (M x N)
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);

    // Initialize output tensor to zero since we will accumulate partial sums
    auto C = torch::zeros({M, N}, A.options());

    // Set tile sizes for output dimensions and K splitting
    const int TILE_SIZE = 16;  // tile size for M and N dimensions
    const int TILE_K = 16;     // tile size for splitting the K dimension

    // Define grid and block dimensions
    dim3 threads(TILE_SIZE, TILE_SIZE);
    // blocks.x covers columns (N), blocks.y covers rows (M), blocks.z splits the K dimension
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE,
                (K + TILE_K - 1) / TILE_K);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_transpose_atomic_kernel", ([&] {
        matmul_transpose_atomic_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K, TILE_K
        );
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Matrix multiplication with transposed inputs using atomic adds for k-split (CUDA)");
}
