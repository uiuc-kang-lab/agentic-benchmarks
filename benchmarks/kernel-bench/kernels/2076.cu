#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

// Kernel that computes a chunk of rows [rowStart, rowEnd) of the output matrix C = A * B,
// where A and B are lower triangular matrices. Each thread computes one element of C.
__global__ void triangular_mm_kernel_chunk(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            int N,
                                            int rowStart,
                                            int rowEnd) {
    int row = rowStart + blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rowEnd && row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.f;
        } else {
            float sum = 0.f;
            for (int k = col; k <= row; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

// The forward function partitions the computation across multiple CUDA streams
// to overlap kernel execution with any potential memory transfers (if applicable).
// Each stream processes a contiguous chunk of rows from the output matrix C.
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

    // Number of streams to use, can be tuned based on the GPU and problem size.
    int numStreams = 4;
    int rowsPerChunk = (N + numStreams - 1) / numStreams;

    std::vector<cudaStream_t> streams(numStreams);
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    // Launch kernels for each row chunk in its own stream
    for (int i = 0; i < numStreams; ++i) {
        int rowStart = i * rowsPerChunk;
        int rowEnd = std::min(rowStart + rowsPerChunk, N);
        if (rowStart >= rowEnd) break;
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
                  ((rowEnd - rowStart) + TILE_SIZE - 1) / TILE_SIZE);
        triangular_mm_kernel_chunk<<<grid, threadsPerBlock, 0, streams[i]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            rowStart,
            rowEnd
        );
    }

    // Synchronize all streams to ensure computation is complete
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Pipelined triangular matrix multiplication with overlapping computation using CUDA streams");
}
