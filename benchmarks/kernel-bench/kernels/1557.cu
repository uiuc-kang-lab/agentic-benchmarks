#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define BLOCK_SIZE 16

__global__ void shared_memory_optimized_upper_triangular_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        if (row < N && t * BLOCK_SIZE + threadIdx.x < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + t * BLOCK_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (t * BLOCK_SIZE + threadIdx.y < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * BLOCK_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        if (row <= col && row < N && col < N) {
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                if (t * BLOCK_SIZE + k >= row && t * BLOCK_SIZE + k <= col) {
                    sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                }
            }
        }

        __syncthreads();
    }

    if (row <= col && row < N && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor shared_memory_optimized_upper_triangular(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    shared_memory_optimized_upper_triangular_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_memory_optimized_upper_triangular, "Shared memory optimized upper triangular matrix multiplication");
}