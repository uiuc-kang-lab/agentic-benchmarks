#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 64
#define NUM_STREAMS 4

__global__ void matmul_kernel(const float* A, const float* B, float* C, 
                            const int M, const int K, const int N, const int chunk_offset) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_SIZE + ty + chunk_offset;
    const int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        if (row < M && (t * TILE_SIZE + tx) < K)
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (col < N && (t * TILE_SIZE + ty) < K)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(B);
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    torch::Tensor C = torch::zeros({M, N}, A.options());
    
    // Create CUDA streams
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    const int chunk_size = (M + NUM_STREAMS - 1) / NUM_STREAMS;
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        int chunk_offset = i * chunk_size;
        int current_chunk_size = std::min(chunk_size, M - chunk_offset);
        
        if (current_chunk_size <= 0) break;
        
        dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                    (current_chunk_size + TILE_SIZE - 1) / TILE_SIZE);
        
        matmul_kernel<<<blocks, threads, 0, streams[i]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, K, N,
            chunk_offset
        );
    }
    
    // Synchronize and destroy streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with streams (CUDA)");
}