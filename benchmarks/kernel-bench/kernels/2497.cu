#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication with transposed inputs and optimized memory access
template <typename scalar_t>
__global__ void matmul_transpose_kernel(
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
        scalar_t sum = 0;
        
        // Ensure aligned access pattern
        #pragma unroll 4
        for (int k = 0; k < K; k++) {
            // Use __ldg for read-only memory access
            sum += __ldg(&A[k * M + row]) * __ldg(&B[col * K + k]);
        }
        
        // Write result
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

    // Calculate grid and block dimensions - ensure alignment
    const int BLOCK_SIZE = 32;  // Increased block size for better occupancy
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