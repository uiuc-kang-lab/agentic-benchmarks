#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define TILE_SIZE as a compile-time constant to experiment with block sizes.
// Typical values can be 16, 32. For H100, TILE_SIZE=32 is a good candidate.
#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

// CUDA kernel for computing C = A * B, where A and B are lower triangular matrices.
// By using shared memory to hold block tiles from A and B, we aim to reduce global memory access latency.
__global__ void triangular_mm_kernel_shared(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            int N) {
    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.f;

    // Loop over A and B tiles to compute C elements.
    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        // Load elements into shared memory.
        if (row < N && (m * TILE_SIZE + threadIdx.x) < N) {
            Asub[threadIdx.y][threadIdx.x] = A[row * N + m * TILE_SIZE + threadIdx.x];
        } else {
            Asub[threadIdx.y][threadIdx.x] = 0.f;
        }
        if ((m * TILE_SIZE + threadIdx.y) < N && col < N) {
            Bsub[threadIdx.y][threadIdx.x] = B[(m * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bsub[threadIdx.y][threadIdx.x] = 0.f;
        }

        __syncthreads();

        // Accumulate partial sums
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = (row >= col) ? sum : 0.f;  // Ensure lower triangular part.
    }
}

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

    triangular_mm_kernel_shared<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Triangular matrix multiplication with shared memory optimization (CUDA)");
}