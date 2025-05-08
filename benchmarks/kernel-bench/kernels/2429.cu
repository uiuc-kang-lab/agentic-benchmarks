#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// Combined Kernel: Efficient shared memory loading and computation
// A is (K x M): element A[k, m] = A[k * M + m]
// B is (N x K): element B[n, k] = B[n * K + k]
// C is (M x N): element C[m, n] = C[m * N + n]
template <typename scalar_t>
__global__ void optimized_matmul_transpose_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int M,
    int N,
    int K) {

    // Compute the tile indices
    int tile_row = blockIdx.y * BLOCK_SIZE;
    int tile_col = blockIdx.x * BLOCK_SIZE;

    // Compute global row and column for C
    int row = tile_row + threadIdx.y;
    int col = tile_col + threadIdx.x;

    __shared__ scalar_t A_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ scalar_t B_tile[BLOCK_SIZE][BLOCK_SIZE];

    scalar_t sum = 0;

    for (int tile_idx = 0; tile_idx < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile_idx) {

        // Load A into shared memory
        int aRow = tile_idx * BLOCK_SIZE + threadIdx.y;
        int aCol = row;
        A_tile[threadIdx.y][threadIdx.x] = (aCol < M && aRow < K) ? A[aRow * M + aCol] : 0.0f;

        // Load B into shared memory
        int bRow = col;
        int bCol = tile_idx * BLOCK_SIZE + threadIdx.x;
        B_tile[threadIdx.y][threadIdx.x] = (bRow < N && bCol < K) ? B[bRow * K + bCol] : 0.0f;

        __syncthreads();

        // Multiply loaded tiles
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write to C
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// PyTorch binding

torch::Tensor optimized_matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    // Dimensions:
    // A: (K x M), B: (N x K), therefore C: (M x N)
    int K = A.size(0);
    int M = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    // Define block dimensions
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "optimized_matmul_transpose_kernel", ([&] {
        optimized_matmul_transpose_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K);
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_matmul_transpose_cuda, "Optimized matrix multiplication with transposed inputs (CUDA)");
}