#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

__global__ void hybrid_stream_triangular_mm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int rowStart,
    int rowEnd
) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    int row = rowStart + blockRow * TILE_SIZE + threadIdx.y;
    int col = blockCol * TILE_SIZE + threadIdx.x;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    if (row < rowEnd && row < N && col < N && row >= col) {
        for (int t = 0; t < ceil((row - col + 1) / static_cast<float>(TILE_SIZE)); ++t) {
            if (t * TILE_SIZE + threadIdx.x <= row - col) {
                As[threadIdx.y][threadIdx.x] = A[row * N + (col + t * TILE_SIZE + threadIdx.x)];
                Bs[threadIdx.y][threadIdx.x] = B[(col + t * TILE_SIZE + threadIdx.y) * N + col];
            } else {
                As[threadIdx.y][threadIdx.x] = 0.0f;
                Bs[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            __syncthreads();

            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            }
            
            __syncthreads();
        }
        
        C[row * N + col] = sum;
    } else if (row < rowEnd && row < N && col < N && row < col) {
        C[row * N + col] = 0.0f;
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

    const int numStreams = 4;
    int rowsPerChunk = (N + numStreams - 1) / numStreams;

    std::vector<cudaStream_t> streams(numStreams);
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);

    for (int i = 0; i < numStreams; ++i) {
        int rowStart = i * rowsPerChunk;
        int rowEnd = std::min(rowStart + rowsPerChunk, N);
        if (rowStart >= rowEnd) break;

        dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE,
                      ((rowEnd - rowStart) + TILE_SIZE - 1) / TILE_SIZE);

        hybrid_stream_triangular_mm_kernel<<<numBlocks, threadsPerBlock, 0, streams[i]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            rowStart,
            rowEnd
        );
    }

    for (int i = 0; i < numStreams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}