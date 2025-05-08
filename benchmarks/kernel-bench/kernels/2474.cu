#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication with transposed inputs using optimized shared memory
template <typename scalar_t>
__global__ void matmul_transpose_shared_mem_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    
    // Define block size and shared memory for tiles
    const int TILE_SIZE = 32;
    __shared__ scalar_t tileA[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t tileB[TILE_SIZE][TILE_SIZE];

    // Calculate position
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    scalar_t sum = 0;
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            tileA[threadIdx.y][threadIdx.x] = A[(t * TILE_SIZE + threadIdx.x) * M + row];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            tileB[threadIdx.y][threadIdx.x] = B[col * K + t * TILE_SIZE + threadIdx.y];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        // Compute using tiles
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_transpose_shared_mem_cuda(torch::Tensor A, torch::Tensor B) {
    // Get dimensions
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);

    // Create output tensor
    auto C = torch::empty({M, N}, A.options());

    // Calculate grid and block dimensions
    const int TILE_SIZE = 32;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_shared_mem_kernel", ([&] {
        matmul_transpose_shared_mem_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_shared_mem_cuda, "Matrix multiplication with transpose and optimized shared memory (CUDA)");
}