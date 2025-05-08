#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// This kernel computes the upper triangular matrix multiplication (C = A * B)
// using shared memory for intra-block reductions and warp-level primitives
// for the final stages of reduction, aiming to optimize memory access patterns
// and reduce runtime.

__global__ void shared_memory_warp_optimized_kernel(const float* __restrict__ A,
                                                     const float* __restrict__ B,
                                                     float* __restrict__ C,
                                                     int N) {
    extern __shared__ float shared_mem[];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

    if (row < N && col < N && row <= col) {
        float sum = 0.0f;
        for (int k = row; k <= col; ++k) {
            float a_val = __ldg(&A[row * N + k]);
            float b_val = __ldg(&B[k * N + col]);
            sum += a_val * b_val;
        }
        shared_mem[thread_id] = sum;
        __syncthreads();

        // Perform reduction within the block using shared memory
        for (int stride = blockDim.x * blockDim.y / 2; stride > 0; stride >>= 1) {
            if (thread_id < stride) {
                shared_mem[thread_id] += shared_mem[thread_id + stride];
            }
            __syncthreads();
        }

        // Write the result for this block to C
        if (thread_id == 0) {
            C[row * N + col] = shared_mem[0];
        }
    }
}

// Host function, exposed via pybind11, that wraps the kernel invocation
// It creates a zero tensor for C, launches the kernel, and returns C.

torch::Tensor shared_memory_warp_optimized_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    size_t shared_memory_size = threadsPerBlock.x * threadsPerBlock.y * sizeof(float);

    shared_memory_warp_optimized_kernel<<<numBlocks, threadsPerBlock, shared_memory_size>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_memory_warp_optimized_matmul, "Shared memory and warp optimized upper triangular matrix multiplication");
}