#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define NUM_STREAMS 4

__global__ void pipelined_tril_mm_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int N,
                                       int stream_offset,
                                       int stream_size) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x * TILE_SIZE;
    int by = (blockIdx.y + stream_offset) * TILE_SIZE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by + ty;
    int col = bx + tx;
    
    float sum = 0.0f;
    
    if (row >= col && row < min(N, stream_offset * TILE_SIZE + stream_size) && col < N) {
        for (int tile = col / TILE_SIZE; tile <= row / TILE_SIZE; ++tile) {
            if ((tile * TILE_SIZE + tx) <= row && (by + ty) < N) {
                As[ty][tx] = A[row * N + (tile * TILE_SIZE + tx)];
            } else {
                As[ty][tx] = 0.0f;
            }
            
            if ((tile * TILE_SIZE + ty) <= N && col < N) {
                Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
            } else {
                Bs[ty][tx] = 0.0f;
            }
            
            __syncthreads();
            
            for (int k = 0; k < TILE_SIZE; ++k) {
                if ((tile * TILE_SIZE + k) >= col && (tile * TILE_SIZE + k) <= row) {
                    sum += As[ty][k] * Bs[k][tx];
                }
            }
            
            __syncthreads();
        }
        
        if (row < N && col < N) {
            C[row * N + col] = sum;
        }
    } else if (row < N && col < N) {
        C[row * N + col] = 0.0f;
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

    int N = A.size(0);
    auto C = torch::empty_like(A);
    
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    int rows_per_stream = (N + NUM_STREAMS - 1) / NUM_STREAMS;
    rows_per_stream = ((rows_per_stream + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;

    for (int i = 0; i < NUM_STREAMS; i++) {
        int stream_offset = i * (rows_per_stream / TILE_SIZE);
        int stream_size = min(rows_per_stream, N - stream_offset * TILE_SIZE);
        
        if (stream_size > 0) {
            dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                          (stream_size + TILE_SIZE - 1) / TILE_SIZE);

            pipelined_tril_mm_kernel<<<numBlocks, threadsPerBlock, 0, streams[i]>>>(
                A.data_ptr<float>(),
                B.data_ptr<float>(),
                C.data_ptr<float>(),
                N,
                stream_offset,
                stream_size
            );
        }
    }

    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Pipelined triangular matrix multiplication (CUDA)");
}