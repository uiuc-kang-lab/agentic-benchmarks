#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// This kernel computes the upper triangular matrix multiplication (C = A * B)
// where only elements with row <= col are evaluated. It uses a balanced workload
// distribution by assigning each thread to compute multiple elements in a row-major
// order, ensuring that all threads are utilized effectively.

__global__ void balanced_workload_upper_triangular_kernel(const float* __restrict__ A,
                                                           const float* __restrict__ B,
                                                           float* __restrict__ C,
                                                           int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col_start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int col = col_start; col < N; col += stride) {
        if (row < N && row <= col) {
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
}

// Host function, exposed via pybind11, that wraps the kernel invocation
// It creates a zero tensor for C, launches the kernel, and returns C.

torch::Tensor balanced_workload_upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    balanced_workload_upper_triangular_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &balanced_workload_upper_triangular_matmul, "Balanced workload upper triangular matrix multiplication");
}
