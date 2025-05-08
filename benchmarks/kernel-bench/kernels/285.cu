#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE 16
#define VECTOR_SIZE 4
#define NUM_STREAMS 4

__global__ void bmm_tiled_unrolled_streamed_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int K,
    int N,
    int batch_offset
) {
    int b = blockIdx.z + batch_offset;
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[b * M * K + row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int b_row = t * TILE + threadIdx.y;
        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b * K * N + b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; i += VECTOR_SIZE) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
            sum += As[threadIdx.y][i+1] * Bs[i+1][threadIdx.x];
            sum += As[threadIdx.y][i+2] * Bs[i+2][threadIdx.x];
            sum += As[threadIdx.y][i+3] * Bs[i+3][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[b * M * N + row * N + col] = sum;
    }
}

torch::Tensor forward_bmm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions (K) must match");

    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({batch_size, M, N}, options);

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, 1);

    // Create CUDA streams
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Process batches in chunks using multiple streams
    int batches_per_stream = (batch_size + NUM_STREAMS - 1) / NUM_STREAMS;

    for (int stream_idx = 0; stream_idx < NUM_STREAMS; stream_idx++) {
        int batch_start = stream_idx * batches_per_stream;
        int batch_end = std::min(batch_start + batches_per_stream, batch_size);
        if (batch_start >= batch_size) break;

        grid.z = batch_end - batch_start;

        bmm_tiled_unrolled_streamed_kernel<<<grid, block, 0, streams[stream_idx]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, K, N,
            batch_start
        );
    }

    // Synchronize and cleanup streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Unrolled streamed batched matrix multiplication (CUDA)");
}