#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication with warp-level reduction
template <typename scalar_t>
__global__ void matmul_transpose_kernel_warp_reduce(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {

    // Calculate position
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        scalar_t thread_sum = 0.0;

        for (int k = 0; k < K; k++) {
            // Load elements for this thread
            scalar_t a_val = A[k * M + row];
            scalar_t b_val = B[col * K + k];

            // Accumulate local sum
            thread_sum += a_val * b_val;
        }

        // Reduce within warp (assumes warpSize of 32)
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }

        // Write the reduced result to global memory
        if (threadIdx.x % warpSize == 0) {
            C[row * N + col] = thread_sum;
        }
    }
}

torch::Tensor matmul_transpose_cuda_warp_reduce(torch::Tensor A, torch::Tensor B) {
    // Get dimensions
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);

    // Create output tensor
    auto C = torch::empty({M, N}, A.options());

    // Calculate grid and block dimensions
    const int BLOCK_SIZE = 32;  // optimal for warp efficiency
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_kernel_warp_reduce", ([&] {
        matmul_transpose_kernel_warp_reduce<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda_warp_reduce, "Matrix multiplication with transpose using warp-level primitives (CUDA)");
}