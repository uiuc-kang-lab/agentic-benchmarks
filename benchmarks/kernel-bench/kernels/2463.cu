#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication with transposed inputs
template <typename scalar_t>
__global__ void matmul_transpose_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    
    // Using a tile-based approach with shared memory
    const int BLOCK_SIZE = 32; // Tile dimension
    int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int col = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    
    scalar_t sum = 0;
    
    // Allocate shared memory for tiles of A and B
    __shared__ scalar_t As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ scalar_t Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Loop over tiles along the K dimension
    for (int tile_idx = 0; tile_idx * BLOCK_SIZE < K; tile_idx++) {
        int tiled_k = tile_idx * BLOCK_SIZE;

        // Load one element per thread for matrix A
        // A is stored as (K x M), and accessed as A[k * M + row]
        int a_index = tiled_k + threadIdx.y;
        if (a_index < K && row < M) {
            As[threadIdx.x][threadIdx.y] = A[a_index * M + row];
        } else {
            As[threadIdx.x][threadIdx.y] = 0;
        }

        // Load one element per thread for matrix B
        // B is stored as (N x K), and accessed as B[col * K + k]
        int b_index = tiled_k + threadIdx.x;
        if (b_index < K && col < N) {
            Bs[threadIdx.x][threadIdx.y] = B[col * K + b_index];
        } else {
            Bs[threadIdx.x][threadIdx.y] = 0;
        }

        __syncthreads();

        // Multiply the two tiles
        // Note: each tile is BLOCK_SIZE in the k-dimension
        for (int i = 0; i < BLOCK_SIZE; i++) {
            sum += As[threadIdx.x][i] * Bs[i][threadIdx.y];
        }
        __syncthreads();
    }

    // Write the result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    // Get dimensions
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);

    // Create output tensor
    auto C = torch::empty({M, N}, A.options());

    // Calculate grid and block dimensions
    const int BLOCK_SIZE = 32; // Adjusted block size for optimal performance
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_kernel", ([&] {
        matmul_transpose_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Matrix multiplication with transpose forward (CUDA)");
}