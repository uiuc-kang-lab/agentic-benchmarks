#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>

__global__ void upper_triangular_matmul_kernel_atomic_shared(const float* A, const float* B, float* C, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int linearIndex = blockId * (blockDim.x * blockDim.y) + ty * blockDim.x + tx;

    sharedC[ty * blockDim.x + tx] = 0.0f; // Initialize shared memory

    __syncthreads();

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N && row <= col) {
        float sum = 0.0f;
        for (int k = row; k <= col; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        // Store partial sum in shared memory
        atomicAdd(&sharedC[ty * blockDim.x + tx], sum);
    }

    __syncthreads(); // Ensure all atomic operations are complete

    // Write the final result to global memory only from one thread per position
    if (row < N && col < N && row <= col && tx == 0 && ty == 0) {
        C[row * N + col] = sharedC[ty * blockDim.x + tx];
    }
}

torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);

    auto C = torch::zeros_like(A);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    upper_triangular_matmul_kernel_atomic_shared<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Optimized upper triangular matrix multiplication with atomic shared memory usage");
}
