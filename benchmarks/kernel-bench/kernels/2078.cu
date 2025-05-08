#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

#define NUM_STREAMS 2

__global__ void triangular_mm_kernel_buffered(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            const int N,
                                            const int chunk_start,
                                            const int chunk_size) {
    int row = chunk_start + blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < (chunk_start + chunk_size) && row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.f;
        } else {
            float sum = 0.f;
            #pragma unroll 4
            for (int k = col; k <= row; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
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

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    // Create streams and events for synchronization
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t events[NUM_STREAMS];
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
    }

    // Calculate chunk size for better load balancing
    const int chunk_size = (N + NUM_STREAMS - 1) / NUM_STREAMS;
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);

    // Process matrix in chunks using double buffering
    for (int chunk_start = 0; chunk_start < N; chunk_start += chunk_size) {
        int current_chunk_size = std::min(chunk_size, N - chunk_start);
        int stream_idx = (chunk_start / chunk_size) % NUM_STREAMS;
        
        dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE,
                      (current_chunk_size + TILE_SIZE - 1) / TILE_SIZE);

        // Launch kernel for current chunk
        triangular_mm_kernel_buffered<<<numBlocks, threadsPerBlock, 0, streams[stream_idx]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            chunk_start,
            current_chunk_size
        );

        // Record event for synchronization
        cudaEventRecord(events[stream_idx], streams[stream_idx]);
        
        // If we're about to reuse a stream, wait for its previous work to complete
        if (chunk_start + 2 * chunk_size < N) {
            cudaStreamWaitEvent(streams[stream_idx], events[stream_idx]);
        }
    }

    // Synchronize all streams and clean up
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Double buffered triangular matrix multiplication (CUDA)");
}