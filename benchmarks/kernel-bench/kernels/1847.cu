#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define NUM_STREAMS 4
#define CHUNK_SIZE 1024

__global__ void triangular_mm_kernel_pipelined(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               int N,
                                               int chunk_offset) {
    int row = blockIdx.y * blockDim.y + threadIdx.y + chunk_offset;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < min(chunk_offset + CHUNK_SIZE, N) && col <= row && col < N) {
        float sum = 0.f;
        #pragma unroll 4
        for (int k = col; k <= row; ++k) {
            sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
        }
        C[row * N + col] = sum;
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

    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    
    // Process matrix in chunks
    for (int chunk_start = 0; chunk_start < N; chunk_start += CHUNK_SIZE) {
        int current_chunk_size = min(CHUNK_SIZE, N - chunk_start);
        dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                      (current_chunk_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

        int stream_idx = (chunk_start / CHUNK_SIZE) % NUM_STREAMS;

        // Asynchronously launch kernel
        triangular_mm_kernel_pipelined<<<numBlocks, threadsPerBlock, 0, streams[stream_idx]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            chunk_start
        );
    }

    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}