#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 64  // Increased block size for better occupancy and performance

// CUDA kernel for matrix multiplication with transposed inputs using shared memory tiling
// and a configurable block size
template <typename scalar_t>
__global__ void matmul_transpose_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    
    // Each thread computes one element of C
    // Compute global row and column indices
    const int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int col = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // Shared memory tiles for A and B
    __shared__ scalar_t tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ scalar_t tileB[BLOCK_SIZE][BLOCK_SIZE];

    scalar_t sum = 0;
    
    // Loop over tiles of the input matrices along the K dimension
    const int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        // Load tile from A
        int k_index = t * BLOCK_SIZE + threadIdx.y;
        if (k_index < K && row < M) {
            // A is stored in transposed form: A[k, row] is at A[k * M + row]
            tileA[threadIdx.y][threadIdx.x] = A[k_index * M + row];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0;
        }

        // Load tile from B
        int k_index_B = t * BLOCK_SIZE + threadIdx.x;
        if (k_index_B < K && col < N) {
            // B is stored in transposed form: B[col, k] is at B[col * K + k]
            tileB[threadIdx.y][threadIdx.x] = B[col * K + k_index_B];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        // Compute partial product for this tile
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += tileA[k][threadIdx.x] * tileB[threadIdx.y][k];
        }

        __syncthreads();
    }

    // Write the computed value to C
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Entry point for the CUDA extension
torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    // Dimensions: A is (K, M) and B is (N, K), computing C = A^T*B^T, resulting in C of shape (M, N)
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    // Configure grid and block dimensions based on BLOCK_SIZE
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

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
    m.def("forward", &matmul_transpose_cuda, "Matrix multiplication with transposed inputs (CUDA) with block size experiment");
}
