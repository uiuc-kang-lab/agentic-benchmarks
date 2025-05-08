#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define NUM_STREAMS 4

__global__ void streamed_pipelined_triangular_mm_kernel(const float* __restrict__ A,
                                                       const float* __restrict__ B,
                                                       float* __restrict__ C,
                                                       int N,
                                                       int start_row) {
    int row = start_row + blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    if (row >= N || col >= N) return;

    if (row < col) {
        C[row * N + col] = 0.f;
        return;
    }

    float sum = 0.f;
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int m = 0; m < numTiles; ++m) {
        int kA = m * TILE_SIZE + threadIdx.x;
        int kB = m * TILE_SIZE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (kA < N && row >= kA) ? A[row * N + kA] : 0;
        sB[threadIdx.y][threadIdx.x] = (kB < N && kB >= col) ? B[kB * N + col] : 0;

        __syncthreads();

        int kStart = max(col, m * TILE_SIZE);
        int kEnd = min(row + 1, (m + 1) * TILE_SIZE);

        #pragma unroll
        for (int k = kStart - m * TILE_SIZE; k < kEnd - m * TILE_SIZE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        if (m < numTiles - 1) __syncthreads();
    }

    C[row * N + col] = sum;
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    const int N = A.size(0);
    auto C = torch::empty_like(A);

    int chunk = (N + NUM_STREAMS - 1) / NUM_STREAMS;
    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; ++s) {
        cudaStreamCreate(&streams[s]);
    }

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);

    for (int s = 0; s < NUM_STREAMS; ++s) {
        int start_row = s * chunk;
        int rows = ((start_row + chunk) > N) ? (N - start_row) : chunk;
        if (rows <= 0) continue;
        dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);

        streamed_pipelined_triangular_mm_kernel<<<numBlocks, threadsPerBlock, 0, streams[s]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            start_row
        );
    }

    for (int s = 0; s < NUM_STREAMS; ++s) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Streamed and Pipelined Triangular Matrix Multiplication (CUDA)");
}
