#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_SIZE 32

// Declare constant memory for read-only data
__constant__ float d_A[TILE_SIZE * TILE_SIZE];
__constant__ float d_B[TILE_SIZE * TILE_SIZE];

__global__ void constant_memory_optimized_upper_triangular_kernel(const float* __restrict__ A,
                                                                  const float* __restrict__ B,
                                                                  float* __restrict__ C,
                                                                  const int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    if (row < N && col < N && row <= col) {
        for (int t = row / TILE_SIZE * TILE_SIZE; t <= col; t += TILE_SIZE) {
            if (threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE) {
                int shared_row = row;
                int shared_col = t + threadIdx.x;
                d_A[threadIdx.y * TILE_SIZE + threadIdx.x] =
                    (shared_row < N && shared_col < N && shared_row <= shared_col) ?
                    A[shared_row * N + shared_col] : 0.0f;
                
                shared_row = t + threadIdx.y;
                shared_col = col;
                d_B[threadIdx.y * TILE_SIZE + threadIdx.x] =
                    (shared_row < N && shared_col < N) ?
                    B[shared_row * N + shared_col] : 0.0f;
            }

            __syncthreads();

            for (int k = 0; k < TILE_SIZE; ++k) {
                int global_k = t + k;
                if (global_k >= row && global_k <= col && global_k < N) {
                    sum += d_A[threadIdx.y * TILE_SIZE + k] * d_B[k * TILE_SIZE + threadIdx.x];
                }
            }

            __syncthreads();
        }

        C[row * N + col] = sum;
    }
}

torch::Tensor constant_memory_optimized_upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    constant_memory_optimized_upper_triangular_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &constant_memory_optimized_upper_triangular_matmul, "Constant memory optimized upper triangular matrix multiplication");
}