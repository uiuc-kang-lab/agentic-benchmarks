#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

constexpr int NUM_STREAMS = 4;
constexpr int CHUNK_SIZE = 1024;
constexpr int TILE_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 256;

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N,
                                   const int chunk_start,
                                   const int chunk_size) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int chunk_offset = chunk_start * N;
    
    const int tx = threadIdx.x % TILE_SIZE;
    const int ty = threadIdx.x / TILE_SIZE;
    
    if (tid < chunk_size * N) {
        const int local_row = tid / N;
        const int global_row = chunk_start + local_row;
        const int col = tid % N;
        
        if (global_row < N && col < N) {
            if (global_row < col) {
                C[chunk_offset + local_row * N + col] = 0.f;
            } else {
                float sum = 0.f;
                
                for (int t = 0; t < (global_row - col + TILE_SIZE - 1) / TILE_SIZE; ++t) {
                    if (tx + t * TILE_SIZE <= global_row) {
                        As[ty][tx] = A[chunk_offset + local_row * N + (t * TILE_SIZE + tx)];
                        Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
                    }
                    __syncthreads();
                    
                    #pragma unroll 16
                    for (int k = 0; k < TILE_SIZE; ++k) {
                        if (t * TILE_SIZE + k <= global_row && t * TILE_SIZE + k >= col) {
                            sum += As[ty][k] * Bs[k][tx];
                        }
                    }
                    __syncthreads();
                }
                C[chunk_offset + local_row * N + col] = sum;
            }
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
    
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i) {
        TORCH_CHECK(cudaStreamCreate(&streams[i]) == cudaSuccess, 
                   "Failed to create CUDA stream");
    }

    for (int chunk_start = 0; chunk_start < N; chunk_start += CHUNK_SIZE) {
        const int stream_idx = (chunk_start / CHUNK_SIZE) % NUM_STREAMS;
        const int current_chunk_size = std::min(CHUNK_SIZE, N - chunk_start);
        const int numElements = current_chunk_size * N;
        const int numBlocks = (numElements + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

        triangular_mm_kernel<<<numBlocks, MAX_THREADS_PER_BLOCK, 0, streams[stream_idx]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            chunk_start,
            current_chunk_size
        );
    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        TORCH_CHECK(cudaStreamSynchronize(streams[i]) == cudaSuccess,
                   "Stream synchronization failed");
        TORCH_CHECK(cudaStreamDestroy(streams[i]) == cudaSuccess,
                   "Stream destruction failed");
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}