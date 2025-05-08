#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>

const int CHUNK_SIZE = 1024;  // Size of matrix chunks to pipeline
const int NUM_STREAMS = 4;    // Number of concurrent streams

__global__ void upper_triangular_matmul_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int N, int chunk_start) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    row += chunk_start;  // Offset by chunk start
    
    if (row < N && col < N && row <= col) {
        float sum = 0.0f;
        #pragma unroll 4
        for (int k = row; k <= col; k++) {
            sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
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

    dim3 threadsPerBlock(32, 32);
    
    // Process matrix in chunks
    for (int chunk_start = 0; chunk_start < N; chunk_start += CHUNK_SIZE) {
        int stream_idx = (chunk_start / CHUNK_SIZE) % NUM_STREAMS;
        int chunk_size = std::min(CHUNK_SIZE, N - chunk_start);
        
        dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                      (chunk_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

        upper_triangular_matmul_kernel<<<numBlocks, threadsPerBlock, 0, streams[stream_idx]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            chunk_start
        );
    }
    
    // Cleanup streams
    for (auto& stream : streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Pipelined upper triangular matrix multiplication");
}