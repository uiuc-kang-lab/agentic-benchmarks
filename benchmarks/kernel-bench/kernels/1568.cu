#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// This kernel computes the upper triangular matrix multiplication (C = A * B)
// using warp-level primitives to optimize small reductions within a warp.
// It distributes the workload more evenly by having each thread compute multiple
// elements, and uses __shfl_down_sync() to perform reductions within a warp.

__global__ void warp_optimized_upper_triangular_kernel(const float* __restrict__ A,
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
            // Use warp-level primitive to reduce within the warp
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
            }
            // Write the result from the first thread in the warp
            if (threadIdx.x % 32 == 0) {
                C[row * N + col] = sum;
            }
        }
    }
}

// Host function, exposed via pybind11, that wraps the kernel invocation
// It creates a zero tensor for C, launches the kernel, and returns C.

torch::Tensor warp_optimized_upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    dim3 threadsPerBlock(32, 8); // Use 8 warps per block
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    warp_optimized_upper_triangular_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &warp_optimized_upper_triangular_matmul, "Warp optimized upper triangular matrix multiplication");
}
