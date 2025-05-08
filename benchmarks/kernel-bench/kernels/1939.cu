#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define NUM_STREAMS 4

__global__ void optimized_stream_kernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int N,
                                        int chunk_start) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1]; // +1 for bank conflict avoidance
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    int row = chunk_start + blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    if (row >= N || col >= N) return;
    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }
    
    float sum = 0.0f;
    
    int global_k_start = max(col, chunk_start);
    int num_tiles = (min(row, N-1) - global_k_start + TILE_SIZE) / TILE_SIZE;
    
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        int k_tile = global_k_start + tile_idx * TILE_SIZE;
        int tile_k_end = min(k_tile + TILE_SIZE, row + 1);
        
        // Load A tile (coalesced)
        int a_col = k_tile + threadIdx.x;
        if (a_col <= row && a_col < N) {
            As[threadIdx.y][threadIdx.x] = __ldg(&A[row * N + a_col]);
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load B tile (coalesced)
        int b_row = k_tile + threadIdx.y;
        if (b_row < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = __ldg(&B[b_row * N + col]);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        
        int contrib_end = min(tile_k_end - k_tile, TILE_SIZE);
        for (int k = 0; k < contrib_end; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        if (tile_idx < num_tiles - 1) __syncthreads();
    }
    
    C[row * N + col] = sum;
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    int N = A.size(0);
    auto C = torch::empty_like(A);
    
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i)
        cudaStreamCreate(&streams[i]);
    
    int chunk_size = (N + NUM_STREAMS - 1) / NUM_STREAMS;
    chunk_size = ((chunk_size + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    for (int s = 0; s < NUM_STREAMS; ++s) {
        int start_row = s * chunk_size;
        if (start_row >= N) continue;
        
        int end_row = min((s + 1) * chunk_size, N);
        int rows = end_row - start_row;
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);
        
        optimized_stream_kernel<<<grid, block, 0, streams[s]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            start_row
        );
    }
    
    for (int s = 0; s < NUM_STREAMS; ++s) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
    }
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized stream-pipelined tri matmul");
}