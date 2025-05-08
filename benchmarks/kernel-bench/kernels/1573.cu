#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// This kernel computes the upper triangular matrix multiplication (C = A * B)
// where only elements with row <= col are evaluated. It uses shared memory
// to optimize intra-block reductions and warp-level primitives for the final stages.

__global__ void shared_memory_optimized_upper_triangular_kernel(const float* __restrict__ A,
                                                                 const float* __restrict__ B,
                                                                 float* __restrict__ C,
                                                                 int N) {
    extern __shared__ float shared_data[];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col_start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int col = col_start; col < N; col += stride) {
        if (row < N && row <= col) {
            float sum = 0.0f;
            for (int k = row; k <= col; ++k) {
                float a_val = __ldg(&A[row * N + k]);
                float b_val = __ldg(&B[k * N + col]);
                sum += a_val * b_val;
            }
            shared_data[threadIdx.y * blockDim.x + threadIdx.x] = sum;
            __syncthreads();

            // Reduce within the block using warp-level primitives
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
            }

            // Write the result for this block
            if (threadIdx.x % warpSize == 0) {
                atomicAdd(&C[row * N + col], sum);
            }
        }
    }
}

// Host function, exposed via pybind11, that wraps the kernel invocation
// It creates a zero tensor for C, launches the kernel, and returns C.

torch::Tensor shared_memory_optimized_upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    size_t shared_memory_size = threadsPerBlock.x * threadsPerBlock.y * sizeof(float);

    shared_memory_optimized_upper_triangular_kernel<<<numBlocks, threadsPerBlock, shared_memory_size>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_memory_optimized_upper_triangular_matmul, "Shared memory optimized upper triangular matrix multiplication");
}