#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE 16
#define NUM_STREAMS 4
#define VECTOR_SIZE 4

__global__ void bmm_optimized_kernel(
    const float4* __restrict__ A,
    const float4* __restrict__ B,
    float* __restrict__ C,
    const int batch_size,
    const int M,
    const int K,
    const int N,
    const int batch_offset
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z + batch_offset;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = by * TILE + ty;
    const int col = bx * TILE + tx;

    const int batch_offset_a = bz * M * K;
    const int batch_offset_b = bz * K * N;
    
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        if (row < M && t * TILE + tx * VECTOR_SIZE < K) {
            float4 tmp_a = A[batch_offset_a/4 + row * (K/4) + (t * TILE + tx)];
            As[ty][tx] = tmp_a.x;
            if (tx*4 + 1 < TILE) As[ty][tx*4 + 1] = tmp_a.y;
            if (tx*4 + 2 < TILE) As[ty][tx*4 + 2] = tmp_a.z;
            if (tx*4 + 3 < TILE) As[ty][tx*4 + 3] = tmp_a.w;
        } else {
            As[ty][tx] = 0.0f;
        }

        if (t * TILE + ty < K && col * VECTOR_SIZE < N) {
            float4 tmp_b = B[batch_offset_b/4 + (t * TILE + ty) * (N/4) + col];
            Bs[ty][tx*4] = tmp_b.x;
            if (tx*4 + 1 < TILE) Bs[ty][tx*4 + 1] = tmp_b.y;
            if (tx*4 + 2 < TILE) Bs[ty][tx*4 + 2] = tmp_b.z;
            if (tx*4 + 3 < TILE) Bs[ty][tx*4 + 3] = tmp_b.w;
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k += 4) {
            sum += As[ty][k] * Bs[k][tx];
            sum += As[ty][k+1] * Bs[k+1][tx];
            sum += As[ty][k+2] * Bs[k+2][tx];
            sum += As[ty][k+3] * Bs[k+3][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[bz * M * N + row * N + col] = sum;
    }
}

torch::Tensor forward_bmm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions (K) must match");

    const int batch_size = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int N = B.size(2);

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({batch_size, M, N}, options);

    dim3 threads(TILE/4, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, 1);

    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int batches_per_stream = (batch_size + NUM_STREAMS - 1) / NUM_STREAMS;
    
    for (int stream_idx = 0; stream_idx < NUM_STREAMS; stream_idx++) {
        int batch_start = stream_idx * batches_per_stream;
        int batch_end = std::min(batch_start + batches_per_stream, batch_size);
        if (batch_start >= batch_size) break;

        blocks.z = batch_end - batch_start;
        
        bmm_optimized_kernel<<<blocks, threads, 0, streams[stream_idx]>>>(
            reinterpret_cast<const float4*>(A.data_ptr<float>()),
            reinterpret_cast<const float4*>(B.data_ptr<float>()),
            C.data_ptr<float>(),
            batch_size, M, K, N,
            batch_start
        );
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return C;
}