#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// This kernel computes the upper triangular matrix multiplication (C = A * B)
// where only elements with row <= col are evaluated. It optimizes thread and block
// indexing to ensure efficient mapping of threads to the problem domain.

__global__ void optimized_thread_block_indexing_kernel(const float* __restrict__ A,
                                                        const float* __restrict__ B,
                                                        float* __restrict__ C,
                                                        int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N && row <= col) {
        float sum = 0.0f;
        // Loop from k = row to k = col in order to follow the upper triangular property
        for (int k = row; k <= col; ++k) {
            float a_val = __ldg(&A[row * N + k]);
            float b_val = __ldg(&B[k * N + col]);
            sum += a_val * b_val;
        }
        C[row * N + col] = sum;
    }
}

// Host function, exposed via pybind11, that wraps the kernel invocation
// It creates a zero tensor for C, launches the kernel, and returns C.

torch::Tensor optimized_thread_block_indexing_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    // Adjust block size to optimize for the GPU architecture
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    optimized_thread_block_indexing_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_thread_block_indexing_matmul, "Optimized thread and block indexing upper triangular matrix multiplication");
}
