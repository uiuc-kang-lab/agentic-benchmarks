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

template <typename scalar_t>
__global__ void optimized_thread_block_indexing_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int M,
    int N,
    int K) {

    // Calculate position using optimized indexing
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        scalar_t sum = 0;
        for (int k = 0; k < K; k++) {
            // A is transposed: (k,M) -> access A[k * M + row]
            // B is transposed: (N,K) -> access B[col * K + k]
            sum += __ldg(&A[k * M + row]) * __ldg(&B[col * K + k]);
        }
        C[row * N + col] = sum;
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
    dim3 blocks((M + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "optimized_thread_block_indexing_kernel", ([&] {
        optimized_thread_block_indexing_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K);
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Optimized matrix multiplication with transposed inputs using efficient thread and block indexing (CUDA)");
}