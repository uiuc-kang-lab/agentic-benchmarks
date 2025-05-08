#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void matmul_transpose_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    
    const int TILE_SIZE = 32;
    __shared__ scalar_t As[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t Bs[TILE_SIZE][TILE_SIZE];
    
    const int row = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int col = blockIdx.y * TILE_SIZE + threadIdx.y;
    const bool valid_thread = (row < M && col < N);
    
    scalar_t sum = 0;
    
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < num_tiles; ++tile) {
        const int tile_idx = tile * TILE_SIZE;
        const int k_idx = tile_idx + threadIdx.y;
        const int b_idx = tile_idx + threadIdx.x;
        
        // Collaborative loading with fewer divergent branches
        As[threadIdx.x][threadIdx.y] = ((row < M && k_idx < K) ? 
            A[k_idx * M + row] : 0);
        
        Bs[threadIdx.y][threadIdx.x] = ((col < N && b_idx < K) ? 
            B[col * K + b_idx] : 0);
        
        __syncthreads();
        
        // Compute partial dot product for this tile
        if (valid_thread) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += As[threadIdx.x][k] * Bs[threadIdx.y][k];
            }
        }
        
        __syncthreads();
    }
    
    if (valid_thread) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);
    
    auto C = torch::empty({M, N}, A.options());
    
    const int TILE_SIZE = 32;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((M + TILE_SIZE - 1) / TILE_SIZE,
                (N + TILE_SIZE - 1) / TILE_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_kernel", ([&] {
        matmul_transpose_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Matrix multiplication with transpose forward (CUDA)");
}