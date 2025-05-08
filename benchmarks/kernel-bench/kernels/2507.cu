#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tiled kernel for matrix multiplication with transposed inputs.
// The operation performed is: C = A^T * B^T, where
//   A is stored as a (K x M) matrix, accessed as A[k * M + i] for A^T[i,k],
//   B is stored as a (N x K) matrix, accessed as B[j * K + k] for B^T[k,j],
//   and C is (M x N) with C[i * N + j] = sum_{k=0}^{K-1} A[k * M + i] * B[j * K + k].

// Templated kernel with block (tile) size as a template parameter
template <typename scalar_t, int BLOCK_SIZE>
__global__ void tiled_matmul_transpose_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,  // number of rows in C (and A^T)
    const int N,  // number of columns in C (and B^T)
    const int K)  // common dimension
{
    // Each thread computes one element of C: C[i,j]
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;  // row index in C
    int j = blockIdx.y * BLOCK_SIZE + threadIdx.y;  // column index in C
    
    scalar_t sum = 0;

    // Declare shared memory tiles for A^T and B^T
    __shared__ scalar_t s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ scalar_t s_B[BLOCK_SIZE][BLOCK_SIZE];

    // Loop over tiles along the k dimension
    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int t = 0; t < numTiles; t++) {
        // Compute the global k index for loading elements
        int k_index_A = t * BLOCK_SIZE + threadIdx.y;  // for loading A: A is accessed as A[k, i] for A^T[i,k]
        if (i < M && k_index_A < K)
            s_A[threadIdx.x][threadIdx.y] = A[k_index_A * M + i];
        else
            s_A[threadIdx.x][threadIdx.y] = 0;

        int k_index_B = t * BLOCK_SIZE + threadIdx.x;  // for loading B: B is accessed as B[j, k] for B^T[k,j]
        if (j < N && k_index_B < K)
            s_B[threadIdx.x][threadIdx.y] = B[j * K + k_index_B];
        else
            s_B[threadIdx.x][threadIdx.y] = 0;

        __syncthreads();

        // Accumulate partial products
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += s_A[threadIdx.x][k] * s_B[k][threadIdx.y];
        }

        __syncthreads();
    }

    // Write result to global memory
    if (i < M && j < N)
        C[i * N + j] = sum;
}

// Host function with tunable block size (tile size).
// The blockSize parameter represents the tile dimension. For example:
//   blockSize = 8  -> 8x8 = 64 threads per block
//   blockSize = 16 -> 16x16 = 256 threads per block
//   blockSize = 32 -> 32x32 = 1024 threads per block
// You can experiment with these values to determine the optimal configuration on the H100.

torch::Tensor matmul_transpose_cuda_tiled(torch::Tensor A, torch::Tensor B, int blockSize) {
    // A: shape (K, M) and B: shape (N, K), C: shape (M, N)
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);
    
    auto C = torch::empty({M, N}, A.options());

    // Set up grid and block dimensions based on the provided blockSize (tile dimension)
    dim3 threads(blockSize, blockSize);
    dim3 blocks((M + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);

    // Dispatch kernel based on the blockSize
    if (blockSize == 8) {
        AT_DISPATCH_FLOATING_TYPES(A.type(), "tiled_matmul_transpose_kernel", ([&] {
            tiled_matmul_transpose_kernel<scalar_t,8><<<blocks, threads>>>(
                A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(),
                C.data_ptr<scalar_t>(),
                M, N, K);
        }));
    } else if (blockSize == 16) {
        AT_DISPATCH_FLOATING_TYPES(A.type(), "tiled_matmul_transpose_kernel", ([&] {
            tiled_matmul_transpose_kernel<scalar_t,16><<<blocks, threads>>>(
                A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(),
                C.data_ptr<scalar_t>(),
                M, N, K);
        }));
    } else if (blockSize == 32) {
        AT_DISPATCH_FLOATING_TYPES(A.type(), "tiled_matmul_transpose_kernel", ([&] {
            tiled_matmul_transpose_kernel<scalar_t,32><<<blocks, threads>>>(
                A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(),
                C.data_ptr<scalar_t>(),
                M, N, K);
        }));
    } else {
        // Default to blockSize = 16 if unsupported value is provided
        AT_DISPATCH_FLOATING_TYPES(A.type(), "tiled_matmul_transpose_kernel", ([&] {
            tiled_matmul_transpose_kernel<scalar_t,16><<<blocks, threads>>>(
                A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(),
                C.data_ptr<scalar_t>(),
                M, N, K);
        }));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda_tiled, "Tiled and tunable blocksize matrix multiplication with transposed inputs (CUDA)");
}
