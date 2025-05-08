#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define NUM_STREAMS 4

__global__ void triangular_mm_stream_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int N,
    const int start_row) {
    
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Global row and column indices
    const int row = start_row + blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= N || col >= N) return;
    
    // Zero out upper triangular part
    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }
    
    float acc = 0.0f;
    
    // Calculate number of tiles needed
    const int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < num_tiles; tile++) {
        const int tile_start = tile * TILE_SIZE;
        
        // Skip tiles that won't contribute to the result
        if (tile_start > row) break;
        
        // Load tile from A into shared memory
        if (tile_start + threadIdx.x <= row && tile_start + threadIdx.x < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + tile_start + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile from B into shared memory
        if (tile_start + threadIdx.y < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(tile_start + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            int global_k = tile_start + k;
            if (global_k <= row && global_k >= col && global_k < N) {
                acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            }
        }
        
        __syncthreads();
    }
    
    C[row * N + col] = acc;
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Matrices must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Matrices must have same dimensions");
    
    const int N = A.size(0);
    auto C = torch::empty_like(A);
    
    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    const int rows_per_stream = (N + NUM_STREAMS - 1) / NUM_STREAMS;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    
    // Launch kernels in different streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        int start_row = i * rows_per_stream;
        int stream_rows = min(rows_per_stream, N - start_row);
        if (stream_rows <= 0) continue;
        
        dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                    (stream_rows + TILE_SIZE - 1) / TILE_SIZE);
        
        triangular_mm_stream_kernel<<<blocks, threads, 0, streams[i]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            start_row
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
    m.def("forward", &forward, "Pipelined triangular matrix multiplication (CUDA)");
}