#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

// CUDA kernel for lower-triangular matrix multiplication with manually unrolled inner loop
__global__ void triangular_mm_kernel_unrolled(const float* __restrict__ A,
                                                const float* __restrict__ B,
                                                float* __restrict__ C,
                                                int N) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row < N && col < N) {
        // For upper triangular elements, output zero
        if (row < col) {
            C[row * N + col] = 0.f;
        } else {
            float sum = 0.f;
            int total_iters = row - col + 1;  // number of elements to sum
            int k = col;
            int unroll_iters = total_iters / 4;
            int remainder = total_iters % 4;

            // Unroll the loop by a factor of 4
            #pragma unroll
            for (int i = 0; i < unroll_iters; ++i) {
                sum += A[row * N + k]     * B[k * N + col];
                sum += A[row * N + k + 1] * B[(k + 1) * N + col];
                sum += A[row * N + k + 2] * B[(k + 2) * N + col];
                sum += A[row * N + k + 3] * B[(k + 3) * N + col];
                k += 4;
            }

            // Process any remaining iterations
            #pragma unroll
            for (int i = 0; i < remainder; ++i) {
                sum += A[row * N + k] * B[k * N + col];
                k++;
            }

            C[row * N + col] = sum;
        }
    }
}

// C++ interface exposed to PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_kernel_unrolled<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Unrolled triangular matrix multiplication (CUDA)");
}
