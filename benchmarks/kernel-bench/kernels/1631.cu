#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>

#define NUM_STREAMS 4
#define CHUNK_SIZE 1024

__global__ void upper_triangular_matmul_kernel(const float* A, const float* B, float* C, 
                                             int N, int chunk_start, int chunk_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x + chunk_start;
    
    if (row < N && col < N && row <= col && col < (chunk_start + chunk_size)) {
        float sum = 0.0f;
        #pragma unroll 4
        for (int k = row; k <= col; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);
    
    // Create CUDA streams
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 threadsPerBlock(16, 32);
    
    // Process matrix in chunks using different streams
    for (int chunk_start = 0; chunk_start < N; chunk_start += CHUNK_SIZE) {
        int chunk_size = min(CHUNK_SIZE, N - chunk_start);
        int stream_idx = (chunk_start / CHUNK_SIZE) % NUM_STREAMS;
        
        dim3 numBlocks((chunk_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                      (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
        
        upper_triangular_matmul_kernel<<<numBlocks, threadsPerBlock, 0, streams[stream_idx]>>>(
            A.data_ptr<float>(), 
            B.data_ptr<float>(), 
            C.data_ptr<float>(), 
            N,
            chunk_start,
            chunk_size
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
    m.def("forward", &upper_triangular_matmul, "Streamed upper triangular matrix multiplication");
}