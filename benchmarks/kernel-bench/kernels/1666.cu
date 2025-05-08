#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>

// Kernel for upper triangular matrix multiplication using shared memory optimization

__global__ void upper_triangular_matmul_kernel(const float* __restrict__ A,
                                                 const float* __restrict__ B,
                                                 float* __restrict__ C,
                                                 int N) {
    __shared__ float shared_A[32][32];
    __shared__ float shared_B[32][32];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N && row <= col) {
        float sum = 0.0f;

        for (int tile = 0; tile < (N + 32 - 1) / 32; ++tile) {
            if (tile * 32 + threadIdx.x < N && row < N) {
                shared_A[threadIdx.y][threadIdx.x] = A[row * N + tile * 32 + threadIdx.x];
            } else {
                shared_A[threadIdx.y][threadIdx.x] = 0.0f;
            }

            if (tile * 32 + threadIdx.y < N && col < N) {
                shared_B[threadIdx.y][threadIdx.x] = B[(tile * 32 + threadIdx.y) * N + col];
            } else {
                shared_B[threadIdx.y][threadIdx.x] = 0.0f;
            }

            __syncthreads(); // Synchronize threads within a block to ensure the shared memory is loaded

            #pragma unroll
            for (int k = 0; k < 32; ++k) {
                sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
            }

            if (tile * 32 + 31 >= col) break; // Avoid unnecessary synchronization beyond the block scope

            __syncthreads(); // Ensure all threads reach this point before loading new data
        }

        C[row * N + col] = sum;
    }
}

// Host function that wraps the kernel call
torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    upper_triangular_matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Optimized upper triangular matrix multiplication with shared memory");
}
