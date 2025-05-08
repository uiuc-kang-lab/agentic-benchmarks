#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>

// Optimized kernel for upper triangular matrix multiplication
// Distributes workload evenly across threads and blocks

__global__ void balanced_upper_triangular_matmul_kernel(const float* __restrict__ A,
                                                         const float* __restrict__ B,
                                                         float* __restrict__ C,
                                                         int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Only compute upper triangular part
    if (row < N && col < N && row <= col) {
        float sum = 0.0f;

        // Instead of looping over all k, we only need to sum over k from row to col
        int start_k = row;
        int end_k = col;

        for (int k = start_k; k <= end_k; ++k) {
            float a_val = __ldg(&A[row * N + k]);
            float b_val = __ldg(&B[k * N + col]);
            sum += a_val * b_val;
        }
        
        C[row * N + col] = sum;
    }
}

// Host function that wraps the kernel call
torch::Tensor balanced_upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    // Adjust block size to balance workload
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    balanced_upper_triangular_matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &balanced_upper_triangular_matmul, "Balanced upper triangular matrix multiplication");
}