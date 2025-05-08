#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

// Optimized kernel for upper triangular matrix multiplication (C = A * B)
// This kernel combines the efficient workload distribution of the first kernel
// with the simplicity of the second kernel.
__global__ void optimized_upper_triangular_kernel(const float* __restrict__ A,
                                                   const float* __restrict__ B,
                                                   float* __restrict__ C,
                                                   int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col_start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (row < N) {
        for (int col = col_start; col < N; col += stride) {
            if (row <= col) {
                float sum = 0.0f;
                // Loop from k = row to k = col to leverage upper triangular property
                for (int k = row; k <= col; ++k) {
                    sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
                }
                C[row * N + col] = sum;
            }
        }
    }
}

// Host function to wrap the kernel invocation
// It initializes the output tensor C and launches the optimized kernel.
torch::Tensor optimized_upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    // Using a block size that balances load and reduces divergence
    dim3 threadsPerBlock(32, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    optimized_upper_triangular_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_upper_triangular_matmul, "Optimized upper triangular matrix multiplication");
}