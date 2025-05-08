#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication with transposed inputs and coalesced memory access
template <typename scalar_t>
__global__ void matmul_transpose_coalesced_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    
    // Calculate position
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    // Shared memory for tiles
    __shared__ scalar_t tileA[16][16];
    __shared__ scalar_t tileB[16][16];

    scalar_t sum = 0;
    for (int t = 0; t < (K + 16 - 1) / 16; ++t) {
        // Load tiles into shared memory with coalesced access
        tileA[threadIdx.y][threadIdx.x] = (t * 16 + threadIdx.y < K && row < M) ? A[(t * 16 + threadIdx.y) * M + row] : 0.0;

        if (t * 16 + threadIdx.x < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[col * K + t * 16 + threadIdx.x];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < 16; ++k) {
            sum += tileA[k][threadIdx.x] * tileB[threadIdx.y][k];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_transpose_coalesced_cuda(torch::Tensor A, torch::Tensor B) {
    // Get dimensions
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);

    // Create output tensor
    auto C = torch::empty({M, N}, A.options());

    // Calculate grid and block dimensions
    const int BLOCK_SIZE = 16;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_coalesced_kernel", ([&] {
        matmul_transpose_coalesced_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_coalesced_cuda, "Matrix multiplication with transpose and coalesced access forward (CUDA)");
}
