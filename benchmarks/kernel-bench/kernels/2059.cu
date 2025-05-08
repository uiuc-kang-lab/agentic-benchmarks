#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define NUM_STREAMS 4

__global__ void triangular_mm_tiled_kernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int N,
                                         int tile_row,
                                         int tile_col) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Global matrix indices
    int global_row = tile_row * TILE_DIM + row;
    int global_col = tile_col * TILE_DIM + col;
    
    float sum = 0.0f;
    
    if (global_row < N && global_col < N) {
        if (global_row < global_col) {
            C[global_row * N + global_col] = 0.0f;
        } else {
            // Process the matrix in tiles
            for (int t = tile_col; t <= tile_row && (t * TILE_DIM) <= global_row; ++t) {
                // Load tile data into shared memory
                if (row < TILE_DIM && col < TILE_DIM) {
                    int a_idx = global_row * N + (t * TILE_DIM + col);
                    int b_idx = (t * TILE_DIM + row) * N + global_col;
                    As[threadIdx.y][threadIdx.x] = (a_idx < N * N) ? A[a_idx] : 0.0f;
                    Bs[threadIdx.y][threadIdx.x] = (b_idx < N * N) ? B[b_idx] : 0.0f;
                }
                
                __syncthreads();
                
                // Compute partial sum for this tile
                for (int k = 0; k < TILE_DIM; ++k) {
                    int global_k = t * TILE_DIM + k;
                    if (global_k <= global_row && global_k >= global_col) {
                        sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                    }
                }
                
                __syncthreads();
            }
            
            C[global_row * N + global_col] = sum;
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    
    int N = A.size(0);
    auto C = torch::empty_like(A);
    
    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }
    
    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 numBlocks((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
    
    // Number of tiles
    int num_tiles = (N + TILE_DIM - 1) / TILE_DIM;
    
    // Process tiles using multiple streams
    for (int tile_row = 0; tile_row < num_tiles; ++tile_row) {
        for (int tile_col = 0; tile_col <= tile_row; ++tile_col) {
            int stream_idx = (tile_row * num_tiles + tile_col) % NUM_STREAMS;
            
            triangular_mm_tiled_kernel<<<numBlocks, threadsPerBlock, 0, streams[stream_idx]>>>(
                A.data_ptr<float>(),
                B.data_ptr<float>(),
                C.data_ptr<float>(),
                N,
                tile_row,
                tile_col
            );
        }
    }
    
    // Synchronize and destroy streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Streamed tiled triangular matrix multiplication (CUDA)");
}