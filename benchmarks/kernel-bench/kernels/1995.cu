#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_STREAMS 4

template <int BLOCK_SIZE>
__global__ void coalesced_memory_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int start_row,
    int end_row
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y + start_row;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= end_row || col >= N || row < col) return;

    __shared__ float B_tile[BLOCK_SIZE][BLOCK_SIZE+1];

    float sum = 0.0f;
    const int steps = (row - col + BLOCK_SIZE) / BLOCK_SIZE;

    for (int t = 0; t < steps; ++t) {
        int k = col + t * BLOCK_SIZE + threadIdx.y;
        if (k <= row) {
            B_tile[threadIdx.y][threadIdx.x] = B[k * N + col];
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            int k_tile = col + t * BLOCK_SIZE + i;
            if (k_tile <= row && col + threadIdx.x < N) {
                sum += A[row * N + k_tile] * B_tile[i][threadIdx.x];
            }
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}

at::Tensor forward_optimized(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    const int N = A.size(0);
    auto C = torch::empty_like(A);

    const int BLOCK_SIZE = 16;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    cudaStream_t streams[NUM_STREAMS];

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    const int chunk_size = (N + NUM_STREAMS - 1) / NUM_STREAMS;

    for (int i = 0; i < NUM_STREAMS; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, N);
        if (start >= end) continue;

        dim3 grid(
            (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (end - start + BLOCK_SIZE - 1) / BLOCK_SIZE
        );

        coalesced_memory_kernel<BLOCK_SIZE><<<grid, block, 0, streams[i]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            start,
            end
        );
    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_optimized, "Memory-coalesced triangular matmul (CUDA)");
}
