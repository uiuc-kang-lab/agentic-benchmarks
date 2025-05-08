#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Tunable parameters
constexpr int TILE_DIM = 32;
constexpr int NUM_STREAMS = 4;
constexpr int CHUNK_SIZE = 1024;

__global__ void triangular_mm_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int N,
    const int chunk_start,
    const int chunk_size) {
    
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int chunk_offset = chunk_start * N;
    
    const int row = chunk_start + (tid / N);
    const int col = tid % N;
    
    if (row >= N || col >= N) return;
    
    float sum = 0.0f;
    
    // Process tiles
    for (int t = 0; t < (N + TILE_DIM - 1) / TILE_DIM; ++t) {
        const int tile_start = t * TILE_DIM;
        
        // Collaborative loading of tiles into shared memory
        if (threadIdx.x < TILE_DIM) {
            const int r = row;
            const int c = tile_start + threadIdx.x;
            As[threadIdx.y][threadIdx.x] = (c < N && r < N) ? 
                A[chunk_offset + (row - chunk_start) * N + c] : 0.0f;
            
            const int br = tile_start + threadIdx.y;
            const int bc = col;
            Bs[threadIdx.y][threadIdx.x] = (br < N && bc < N) ? 
                B[br * N + bc] : 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial results for this tile
        if (row >= col) {
            #pragma unroll 8
            for (int k = 0; k < TILE_DIM; ++k) {
                if (tile_start + k <= row) {
                    sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row >= N || col >= N) return;
    
    if (row < col) {
        C[chunk_offset + (row - chunk_start) * N + col] = 0.0f;
    } else {
        C[chunk_offset + (row - chunk_start) * N + col] = sum;
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

    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    
    for (int chunk_start = 0; chunk_start < N; chunk_start += CHUNK_SIZE) {
        const int stream_idx = (chunk_start / CHUNK_SIZE) % NUM_STREAMS;
        const int current_chunk_size = std::min(CHUNK_SIZE, N - chunk_start);
        const int numElements = current_chunk_size * N;
        const dim3 numBlocks((N + TILE_DIM - 1) / TILE_DIM, 
                           (current_chunk_size + TILE_DIM - 1) / TILE_DIM);

        triangular_mm_tiled_kernel<<<numBlocks, threadsPerBlock, 0, streams[stream_idx]>>>(
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