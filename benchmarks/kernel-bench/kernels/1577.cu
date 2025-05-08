#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// This kernel computes the upper triangular matrix multiplication (C = A * B)
// using shared memory and warp-level primitives to optimize reductions within a block.

__global__ void shared_memory_warp_optimized_kernel(const float* __restrict__ A,
                                                     const float* __restrict__ B,
                                                     float* __restrict__ C,
                                                     int N) {
    extern __shared__ float s_data[];  // Shared memory for reduction
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    float threadSum = 0.0f;
    if (row < N && col < N && row <= col) {
        // Use each thread to calculate partial sums
        for (int k = row; k <= col; k += blockDim.x) {
            if (k < N) {
                float a_val = __ldg(&A[row * N + k]);
                float b_val = __ldg(&B[k * N + col]);
                threadSum += a_val * b_val;
            }
        }
        // Store partial sum in shared memory
        s_data[tid] = threadSum;
        __syncthreads();

        // Perform reduction in shared memory
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_data[tid] += s_data[tid + s];
            }
            __syncthreads();
        }

        // Use warp-level primitives to finalize reduction
        if (tid < 32) {
            float val = s_data[tid];
            #pragma unroll
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                val += __shfl_down_sync(0xFFFFFFFF, val, offset);
            }
            // Write the result
            if (tid == 0) {
                C[row * N + col] = val;
            }
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
    size_t sharedMemSize = threadsPerBlock.x * threadsPerBlock.y * sizeof(float);

    shared_memory_warp_optimized_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_memory_warp_optimized_matmul, "Shared memory and warp optimized upper triangular matrix multiplication");
}
