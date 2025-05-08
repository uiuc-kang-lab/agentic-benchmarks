#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHUNK_SIZE 1024
#define NUM_STREAMS 4

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N,
                                   int chunk_start) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    row += chunk_start;  // Offset by chunk start
    
    if (row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.f;
        } else {
            float sum = 0.f;
            #pragma unroll
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
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Matrices must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Matrices must have same dimensions");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Configure kernel dimensions
    const int threads = 16;
    dim3 threadsPerBlock(threads, threads);

    // Process matrix in chunks using different streams
    for (int chunk_start = 0; chunk_start < N; chunk_start += CHUNK_SIZE) {
        int chunk_size = min(CHUNK_SIZE, N - chunk_start);
        dim3 numBlocks((N + threads - 1) / threads,
                      (chunk_size + threads - 1) / threads);

        int stream_idx = (chunk_start / CHUNK_SIZE) % NUM_STREAMS;
        
        triangular_mm_kernel<<<numBlocks, threadsPerBlock, 0, streams[stream_idx]>>>(
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