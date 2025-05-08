#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHUNK_SIZE 1024
#define NUM_STREAMS 4
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N,
                                   const int chunk_start) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];
    
    int row = blockIdx.y * BLOCK_ROWS + threadIdx.y + chunk_start;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    
    if (row < N && col < N && row >= col) {
        float sum = 0.0f;
        
        // Loop over tiles
        for (int t = col; t <= row; t += TILE_DIM) {
            // Collaborative loading of tiles into shared memory
            if (threadIdx.y < TILE_DIM) {
                int tx = t + threadIdx.x;
                int ty = row - (threadIdx.y % BLOCK_ROWS);
                if (tx < N && ty < N && tx <= ty) {
                    As[threadIdx.y][threadIdx.x] = A[ty * N + tx];
                    Bs[threadIdx.y][threadIdx.x] = B[tx * N + col];
                } else {
                    As[threadIdx.y][threadIdx.x] = 0.0f;
                    Bs[threadIdx.y][threadIdx.x] = 0.0f;
                }
            }
            __syncthreads();
            
            // Compute partial dot product using shared memory
            #pragma unroll
            for (int k = 0; k < TILE_DIM; k++) {
                if ((t + k) <= row && (t + k) >= col) {
                    sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                }
            }
            __syncthreads();
        }
        
        C[row * N + col] = sum;
    } else if (row < N && col < N && row < col) {
        C[row * N + col] = 0.0f;
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Matrices must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Matrices must have same dimensions");

    const int N = A.size(0);
    auto C = torch::empty_like(A);
    
    // Create and store CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Configure kernel dimensions
    dim3 threadsPerBlock(TILE_DIM, BLOCK_ROWS);
    
    // Process matrix in chunks using different streams
    for (int chunk_start = 0; chunk_start < N; chunk_start += CHUNK_SIZE) {
        int chunk_size = min(CHUNK_SIZE, N - chunk_start);
        dim3 numBlocks((N + TILE_DIM - 1) / TILE_DIM,
                      (chunk_size + BLOCK_ROWS - 1) / BLOCK_ROWS);
                      
        int stream_idx = (chunk_start / CHUNK_SIZE) % NUM_STREAMS;
        
        triangular_mm_kernel<<<numBlocks, threadsPerBlock, 0, streams[stream_idx]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            chunk_start
        );
    }

    // Clean up streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}