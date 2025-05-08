#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32  // Increased tile size for better occupancy
#define NUM_STREAMS 4 // Increased number of streams
#define CHUNK_SIZE 2048 // Larger chunks for better parallelism

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N,
                                   int chunk_offset) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y + chunk_offset;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Early exit if we're above the diagonal
    if (row < col || row >= N || col >= N) {
        if (row < N && col < N && row < col) {
            C[row * N + col] = 0.f;
        }
        return;
    }

    float sum = 0.0f;
    
    // Calculate start and end tiles for this thread
    int start_tile = col / TILE_SIZE;
    int end_tile = row / TILE_SIZE;
    
    #pragma unroll 2
    for (int t = start_tile; t <= end_tile; t++) {
        // Collaborative loading of tiles
        if (t*TILE_SIZE + tx <= row) {
            As[ty][tx] = A[row * N + (t*TILE_SIZE + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (t*TILE_SIZE + ty < N) {
            Bs[ty][tx] = B[(t*TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        #pragma unroll 8
        for (int k = 0; k < TILE_SIZE; k++) {
            if ((t*TILE_SIZE + k) >= col && (t*TILE_SIZE + k) <= row) {
                sum += As[ty][k] * Bs[k][tx];
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

    // Create CUDA streams with priority
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreateWithPriority(&streams[i], cudaStreamNonBlocking, -i);
    }

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    
    // Process matrix in chunks with stream-based pipeline
    for (int chunk = 0; chunk < N; chunk += CHUNK_SIZE) {
        int chunk_rows = std::min(CHUNK_SIZE, N - chunk);
        dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE,
                      (chunk_rows + TILE_SIZE - 1) / TILE_SIZE);
        
        int stream_idx = (chunk / CHUNK_SIZE) % NUM_STREAMS;
        
        triangular_mm_kernel<<<numBlocks, threadsPerBlock, 0, streams[stream_idx]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            chunk
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
    m.def("forward", &forward, "Optimized triangular matrix multiplication (CUDA)");
}