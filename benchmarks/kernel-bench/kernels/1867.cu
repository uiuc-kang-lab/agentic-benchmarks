#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

constexpr int TILE_SIZE = 32;
constexpr int NUM_STREAMS = 4;
constexpr int CHUNK_SIZE = 1024;

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
    
    const int row = blockIdx.y * TILE_SIZE + ty + chunk_start;
    const int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    if (row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.0f;
            return;
        }
        
        for (int t = 0; t < (row - col + TILE_SIZE - 1) / TILE_SIZE; ++t) {
            const int tile_idx = col + t * TILE_SIZE;
            
            if (tile_idx + tx <= row && row < N) {
                As[ty][tx] = A[row * N + tile_idx + tx];
            } else {
                As[ty][tx] = 0.0f;
            }
            
            if (tile_idx + ty <= row && col < N) {
                Bs[ty][tx] = B[(tile_idx + ty) * N + col];
            } else {
                Bs[ty][tx] = 0.0f;
            }
            
            __syncthreads();
            
            #pragma unroll 8
            for (int k = 0; k < TILE_SIZE; ++k) {
                if (tile_idx + k <= row) {
                    sum += As[ty][k] * Bs[k][tx];
                }
            }
            
            __syncthreads();
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

    const int N = A.size(0);
    auto C = torch::empty_like(A);
    
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 threadsPerBlock(TILE_SIZE * TILE_SIZE);
    
    for (int chunk_start = 0; chunk_start < N; chunk_start += CHUNK_SIZE) {
        const int stream_idx = (chunk_start / CHUNK_SIZE) % NUM_STREAMS;
        const int current_chunk_size = std::min(CHUNK_SIZE, N - chunk_start);
        
        dim3 numBlocks(
            (N + TILE_SIZE - 1) / TILE_SIZE,
            (current_chunk_size + TILE_SIZE - 1) / TILE_SIZE
        );

        triangular_mm_kernel<<<numBlocks, threadsPerBlock, 0, streams[stream_idx]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            chunk_start,
            current_chunk_size
        );
    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}