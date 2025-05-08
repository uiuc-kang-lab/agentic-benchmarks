#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Tunable parameters
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define NUM_STREAMS 4
#define CHUNK_SIZE 1024
#define WARP_SIZE 32

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N,
                                   const int chunk_start) {
    // Calculate thread indices
    const int tid_x = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
    const int tid_y = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;
    const int global_y = chunk_start + tid_y;
    
    // Shared memory for tiling
    __shared__ float As[BLOCK_DIM_Y][BLOCK_DIM_X];
    __shared__ float Bs[BLOCK_DIM_X][BLOCK_DIM_X];
    
    if (tid_x < N && global_y < N) {
        if (global_y < tid_x) {
            C[global_y * N + tid_x] = 0.0f;
        } else {
            float sum = 0.0f;
            
            // Process tiles
            for (int tile = tid_x; tile <= global_y; tile += BLOCK_DIM_X) {
                // Load tile to shared memory
                if (tile + threadIdx.x <= global_y) {
                    As[threadIdx.y][threadIdx.x] = A[global_y * N + tile + threadIdx.x];
                    Bs[threadIdx.y][threadIdx.x] = B[(tile + threadIdx.x) * N + tid_x];
                }
                __syncthreads();
                
                // Compute partial sum for this tile
                #pragma unroll 8
                for (int k = 0; k < BLOCK_DIM_X && (tile + k) <= global_y; k++) {
                    sum += As[threadIdx.y][k] * Bs[threadIdx.y][k];
                }
                __syncthreads();
            }
            
            C[global_y * N + tid_x] = sum;
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
    
    // Create CUDA streams
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 threadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 numBlocks(
        (N + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
        (CHUNK_SIZE + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y
    );

    // Process matrix in chunks using multiple streams
    for (int chunk_start = 0; chunk_start < N; chunk_start += CHUNK_SIZE) {
        const int stream_idx = (chunk_start / CHUNK_SIZE) % NUM_STREAMS;
        
        triangular_mm_kernel<<<numBlocks, threadsPerBlock, 0, streams[stream_idx]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            chunk_start
        );
    }

    // Synchronize and cleanup streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}