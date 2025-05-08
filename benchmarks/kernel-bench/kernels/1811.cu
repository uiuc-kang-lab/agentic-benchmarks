#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32  // Increased tile size
#define NUM_STREAMS 4 // Increased number of streams
#define CHUNK_SIZE 2048 // Larger chunk size for better occupancy
#define WARP_SIZE 32

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N,
                                   const int chunk_offset) {
    // Use vectorized loads where possible
    __shared__ float As[TILE_SIZE][TILE_SIZE+1]; // +1 for bank conflict avoidance
    __shared__ float Bs[TILE_SIZE][TILE_SIZE+1];
    
    const int row = blockIdx.y * blockDim.y + threadIdx.y + chunk_offset;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    float sum = 0.0f;
    
    // Early exit condition
    if (row >= N || col >= N) return;
    
    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    // Calculate number of tiles needed
    const int num_tiles = (min(row, N-1) - col + TILE_SIZE) / TILE_SIZE;
    
    #pragma unroll 2
    for (int t = 0; t < num_tiles; t++) {
        const int tile_idx = col/TILE_SIZE + t;
        const int tile_offset = tile_idx * TILE_SIZE;

        if (row < N && (tile_offset + tx) <= row) {
            As[ty][tx] = A[row * N + tile_offset + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if ((tile_offset + ty) < N && col < N) {
            Bs[ty][tx] = B[(tile_offset + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        if (row >= col) {
            #pragma unroll 8
            for (int k = 0; k < TILE_SIZE; k++) {
                if ((tile_offset + k) >= col && (tile_offset + k) <= row) {
                    sum = fmaf(As[ty][k], Bs[k][tx], sum);
                }
            }
        }
        
        __syncthreads();
    }
    
    if (row >= col) {
        C[row * N + col] = sum;
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "A and B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    
    cudaHostRegister(A.data_ptr<float>(), N * N * sizeof(float), cudaHostRegisterDefault);
    cudaHostRegister(B.data_ptr<float>(), N * N * sizeof(float), cudaHostRegisterDefault);
    
    for (int chunk = 0; chunk < N; chunk += CHUNK_SIZE) {
        const int chunk_rows = std::min(CHUNK_SIZE, N - chunk);
        dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE,
                      (chunk_rows + TILE_SIZE - 1) / TILE_SIZE);
        
        const int stream_idx = (chunk / CHUNK_SIZE) % NUM_STREAMS;
        
        triangular_mm_kernel<<<numBlocks, threadsPerBlock, 0, streams[stream_idx]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            chunk
        );
    }

    cudaHostUnregister(A.data_ptr<float>());
    cudaHostUnregister(B.data_ptr<float>());
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid chunked triangular matrix multiplication (CUDA)");
}