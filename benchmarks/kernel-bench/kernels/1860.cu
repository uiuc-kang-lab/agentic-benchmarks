#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define NUM_STREAMS 4

__global__ void triangular_mm_kernel_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int N,
    const int chunk_start) {
    
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = chunk_start + blockIdx.y * TILE_SIZE + ty;
    const int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    if (row < N && col < N && row >= col) {
        const int num_tiles = (row - col + TILE_SIZE - 1) / TILE_SIZE;
        
        for (int t = 0; t < num_tiles; t++) {
            const int tile_idx = col + t * TILE_SIZE;
            
            // Load tile into shared memory
            if ((row < N) && (tile_idx + tx) <= row) {
                As[ty][tx] = A[row * N + tile_idx + tx];
            } else {
                As[ty][tx] = 0.0f;
            }
            
            if ((tile_idx + ty) < N && col < N) {
                Bs[ty][tx] = B[(tile_idx + ty) * N + col];
            } else {
                Bs[ty][tx] = 0.0f;
            }
            
            __syncthreads();
            
            // Compute partial sum for this tile
            #pragma unroll 8
            for (int k = 0; k < TILE_SIZE; k++) {
                if ((tile_idx + k) <= row) {
                    sum += As[ty][k] * Bs[k][tx];
                }
            }
            
            __syncthreads();
        }
        
        if (row < N && col < N) {
            if (row < col) {
                C[row * N + col] = 0.0f;
            } else {
                C[row * N + col] = sum;
            }
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Matrices must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Matrices must have same dimensions");

    const int N = A.size(0);
    auto C = torch::empty_like(A);
    
    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    const int chunks_per_stream = (N + NUM_STREAMS - 1) / NUM_STREAMS;
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        const int chunk_start = i * chunks_per_stream;
        const int chunk_end = min(chunk_start + chunks_per_stream, N);
        const int chunk_rows = chunk_end - chunk_start;
        
        if (chunk_rows > 0) {
            const dim3 numBlocks(
                (N + TILE_SIZE - 1) / TILE_SIZE,
                (chunk_rows + TILE_SIZE - 1) / TILE_SIZE
            );
            
            triangular_mm_kernel_tiled<<<numBlocks, threadsPerBlock, 0, streams[i]>>>(
                A.data_ptr<float>(),
                B.data_ptr<float>(),
                C.data_ptr<float>(),
                N,
                chunk_start
            );
        }
    }
    
    // Synchronize and destroy streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}