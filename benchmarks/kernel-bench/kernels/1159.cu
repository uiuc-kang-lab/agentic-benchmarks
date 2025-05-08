#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE 32
#define NUM_STREAMS 4

template <typename scalar_t>
__global__ void optimized_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int N, int M, int K, L,
    int stream_batch_start) {
    
    __shared__ scalar_t As[TILE][TILE];
    __shared__ scalar_t Bs[TILE][TILE];
    
    int n = stream_batch_start + blockIdx.z;
    int row = blockIdx.x * TILE + threadIdx.x;
    int col = blockIdx.y * TILE + threadIdx.y;
    
    scalar_t sum = 0;
    
    for (int t = 0; t < (K + TILE-1)/TILE; ++t) {
        int tiled_k = t * TILE;
        
        // Load A tile
        if (n < N && row < M && (tiled_k + threadIdx.y) < K)
            As[threadIdx.x][threadIdx.y] = A[n*M*K + row*K + tiled_k + threadIdx.y];
        else
            As[threadIdx.x][threadIdx.y] = 0;
        
        // Load B tile
        if ((tiled_k + threadIdx.x) < K && col < L)
            Bs[threadIdx.x][threadIdx.y] = B[(tiled_k + threadIdx.x)*L + col];
        else
            Bs[threadIdx.x][threadIdx.y] = 0;
        
        __syncthreads();
        
        for (int k = 0; k < TILE; ++k)
            sum += As[threadIdx.x][k] * Bs[k][threadIdx.y];
        
        __syncthreads();
    }
    
    if (n < N && row < M && col < L)
        output[n*M*L + row*L + col] = sum;
}

void optimized_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {
    
    int N = A.size(0), M = A.size(1), K = A.size(2);
    int L = B.size(1);
    
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i)
        cudaStreamCreate(&streams[i]);
    
    dim3 grid((M+TILE-1)/TILE, (L+TILE-1)/TILE, 1);
    dim3 block(TILE, TILE);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "optimized_forward", ([&] {
        for (int batch_start = 0; batch_start < N; ++batch_start) {
            int stream_id = batch_start % NUM_STREAMS;
            grid.z = 1;  
            
            optimized_kernel<scalar_t><<<grid, block, 0, streams[stream_id]>>>(
                A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                N, M, K, L,
                batch_start);
        }
    }));
    
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
}

TORCH_LIBRARY(module_fn, m) {
    m.def("forward", &optimized_forward);
}