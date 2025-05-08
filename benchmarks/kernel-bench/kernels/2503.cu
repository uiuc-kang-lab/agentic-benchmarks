#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel for matrix multiplication with transposed inputs
template <typename scalar_t>
__global__ void matmul_transpose_kernel_optimized(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    
    // Calculate position
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        scalar_t sum = 0;
        for (int k = 0; k < K; k++) {
            // A is transposed: (k,M) -> access A[k * M + row]
            // B is transposed: (N,K) -> access B[col * K + k]
            sum += A[k * M + row] * B[col * K + k];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_transpose_cuda_optimized(torch::Tensor A, torch::Tensor B) {
    // Get dimensions
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);

    // Create output tensor
    auto C = torch::empty({M, N}, A.options());

    // Calculate grid and block dimensions
    const int BLOCK_SIZE = 32; // Increased block size for better utilization
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_kernel_optimized", ([&] {
        matmul_transpose_kernel_optimized<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda_optimized, "Optimized matrix multiplication with transpose forward (CUDA)");
}