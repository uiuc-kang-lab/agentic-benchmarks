#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Optimized tile dimensions based on typical GPU architectures
#define TILE_DIM 32
#define BLOCK_ROWS 8
#define PREFETCH_COUNT 2

__global__ void optimizedMatMulKernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     const int K, const int M, const int N) {
    // Shared memory for tiling with padding to avoid bank conflicts
    __shared__ float As[TILE_DIM][TILE_DIM + 1];
    __shared__ float Bs[TILE_DIM][TILE_DIM + 1];
    
    const int row = blockIdx.x * BLOCK_ROWS + threadIdx.y;
    const int col = blockIdx.y * TILE_DIM + threadIdx.x;
    
    // Register array for accumulating results
    float results[BLOCK_ROWS] = {0.0f};
    
    // Loop over tiles with prefetching
    for (int tileIdx = 0; tileIdx < (K + TILE_DIM - 1) / TILE_DIM; ++tileIdx) {
        // Collaborative loading of tiles into shared memory
        #pragma unroll
        for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
            if ((tileIdx * TILE_DIM + threadIdx.x) < K && (row + i) < M)
                As[threadIdx.y + i][threadIdx.x] = A[(row + i) * K + (tileIdx * TILE_DIM + threadIdx.x)];
            if ((tileIdx * TILE_DIM + threadIdx.y) < K && col < N)
                Bs[threadIdx.y][threadIdx.x] = B[(tileIdx * TILE_DIM + threadIdx.y) * N + col];
        }
        
        __syncthreads();
        
        // Compute partial results
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            #pragma unroll
            for (int i = 0; i < BLOCK_ROWS; ++i) {
                results[i] += As[i + threadIdx.y][k] * Bs[k][threadIdx.x];
            }
        }
        
        __syncthreads();
    }
    
    // Write results to global memory
    #pragma unroll
    for (int i = 0; i < BLOCK_ROWS; ++i) {
        if ((row + i) < M && col < N) {
            C[(row + i) * N + col] = results[i];
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, 
                "Inputs must be float32");
    
    const int K = A.size(0);
    const int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch");
    const int N = B.size(1);
    
    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));
    
    dim3 threads(TILE_DIM, BLOCK_ROWS);
    dim3 grid((M + BLOCK_ROWS - 1) / BLOCK_ROWS, (N + TILE_DIM - 1) / TILE_DIM);
    
    optimizedMatMulKernel<<<grid, threads>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        K, M, N
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        throw std::runtime_error(cudaGetErrorString(err));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized tiled matrix multiplication (CUDA)");
}