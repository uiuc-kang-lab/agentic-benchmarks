#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_SIZE 32
#define NUM_STREAMS 4
#define STRIP_WIDTH 256

__global__ void stream_pipelined_kernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      const int N,
                                      const int strip_start,
                                      const int strip_end) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = strip_start + blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    if (row < N && col < strip_end && col < N && row <= col) {
        for (int t = row / TILE_SIZE * TILE_SIZE; t <= col; t += TILE_SIZE) {
            if (threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE) {
                int shared_row = row;
                int shared_col = t + threadIdx.x;
                if (shared_row < N && shared_col < N && shared_row <= shared_col) {
                    As[threadIdx.y][threadIdx.x] = A[shared_row * N + shared_col];
                } else {
                    As[threadIdx.y][threadIdx.x] = 0.0f;
                }
                
                shared_row = t + threadIdx.y;
                shared_col = col;
                if (shared_row < N && shared_col < N) {
                    Bs[threadIdx.y][threadIdx.x] = B[shared_row * N + shared_col];
                } else {
                    Bs[threadIdx.y][threadIdx.x] = 0.0f;
                }
            }
            
            __syncthreads();
            
            #pragma unroll 8
            for (int k = 0; k < TILE_SIZE; ++k) {
                int global_k = t + k;
                if (global_k >= row && global_k <= col && global_k < N) {
                    sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                }
            }
            
            __syncthreads();
        }
        
        C[row * N + col] = sum;
    }
}

torch::Tensor stream_pipelined_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);
    
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    
    for (int strip = 0; strip < N; strip += STRIP_WIDTH) {
        int strip_end = min(strip + STRIP_WIDTH, N);
        int strip_width = strip_end - strip;
        
        dim3 numBlocks((strip_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                      (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
        
        int stream_idx = (strip / STRIP_WIDTH) % NUM_STREAMS;
        
        stream_pipelined_kernel<<<numBlocks, threadsPerBlock, 0, streams[stream_idx]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            strip,
            strip_end
        );
    }
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &stream_pipelined_matmul, "Stream pipelined upper triangular matrix multiplication");
}