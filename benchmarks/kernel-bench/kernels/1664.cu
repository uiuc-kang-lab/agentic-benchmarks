#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>

// Optimized kernel for upper triangular matrix multiplication
// This implementation further refines the block and thread mapping for increased efficiency

__global__ void upper_triangular_matmul_kernel(const float* __restrict__ A,
                                                 const float* __restrict__ B,
                                                 float* __restrict__ C,
                                                 int N) {
    // Calculate row and column index from block and thread indices
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int col = blockIdx.y * blockDim.x + threadIdx.x;

    // Only compute upper triangular part by ensuring row index is less than or equal to column index
    if (row < N && col < N && row <= col) {
        float sum = 0.0f;

        // Loop through elements to compute matrix multiplication sum
        for (int k = row; k <= col; k++) {
            sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
        }

        // Write result to output matrix
        C[row * N + col] = sum;
    }
}

// Host function that wraps the kernel call
torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    // Use a different thread block configuration for better utilization
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Launch kernel with modified grid/block configuration
    upper_triangular_matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Upper triangular matrix multiplication with optimized thread/block mapping");
}
