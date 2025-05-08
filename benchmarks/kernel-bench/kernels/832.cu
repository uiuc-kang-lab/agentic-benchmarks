#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define WARP_SIZE 32
#define TILE_DIM 32

__global__ void warp_aligned_matmul_kernel(const float* __restrict__ A, 
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         const int M, const int K, const int N) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_DIM + ty;
    const int col = blockIdx.x * TILE_DIM + tx;
    
    // Pre-compute boundary conditions outside the main loop
    const bool valid_row = row < M;
    const bool valid_col = col < N;
    
    float sum = 0.0f;
    
    // Calculate number of tiles needed
    const int numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    
    for (int tile = 0; tile < numTiles; tile++) {
        const int tileOffset = tile * TILE_DIM;
        
        // Load tile from A using vectorized loads when possible
        if (valid_row && (tileOffset + tx) < K) {
            As[ty][tx] = A[row * K + tileOffset + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile from B using vectorized loads when possible
        if ((tileOffset + ty) < K && valid_col) {
            Bs[ty][tx] = B[(tileOffset + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result using coalesced memory access
    if (valid_row && valid_col) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    // Configure grid and block dimensions to align with warps
    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 numBlocks((N + TILE_DIM - 1) / TILE_DIM,
                   (M + TILE_DIM - 1) / TILE_DIM);
    
    warp_aligned_matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Warp-aligned matrix multiplication (CUDA)");
}