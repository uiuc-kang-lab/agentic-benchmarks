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
    int row = start_row + blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= N || col >= N) return;
    
    // Upper triangular part is zero
    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }
    
    float sum = 0.0f;
    
    // Process the matrix in tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int tile_start = t * TILE_SIZE;
        
        // Load tile from A into shared memory
        if (tile_start + threadIdx.x <= row && tile_start + threadIdx.x < N) {
            As[threadIdx.y][threadIdx.x] = __ldg(&A[row * N + tile_start + threadIdx.x]);
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile from B into shared memory
        if (tile_start + threadIdx.y < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = __ldg(&B[(tile_start + threadIdx.y) * N + col]);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            int global_k = tile_start + k;
            if (global_k <= row && global_k >= col && global_k < N) {
                sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            }
        }
        
        __syncthreads();
    }
    
    C[row * N + col] = sum;
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");
    
    int N = A.size(0);
    auto C = torch::empty_like(A);
    
    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Calculate chunk size for each stream
    int chunk_size = (N + NUM_STREAMS - 1) / NUM_STREAMS;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    
    // Launch kernels in different streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        int start_row = i * chunk_size;
        int rows_in_chunk = min(chunk_size, N - start_row);
        if (rows_in_chunk <= 0) continue;
        
        dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                    (rows_in_chunk + TILE_SIZE - 1) / TILE_SIZE);
        
        triangular_mm_stream_kernel<<<blocks, threads, 0, streams[i]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            start_row
        );
    }
    
    // Synchronize and cleanup streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}