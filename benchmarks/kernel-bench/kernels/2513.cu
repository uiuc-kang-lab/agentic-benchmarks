#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void matmul_transpose_tiled_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    
    const int TILE_SIZE = 32;
    __shared__ scalar_t A_shared[TILE_SIZE][TILE_SIZE + 1];
    __shared__ scalar_t B_shared[TILE_SIZE][TILE_SIZE + 1];
    
    const int row = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int col = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    scalar_t sum = 0;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Vectorized load for A using float2
        if (row < M && (tile * TILE_SIZE + threadIdx.y) < K) {
            reinterpret_cast<float2*>(&A_shared[threadIdx.y][threadIdx.x])[0] = 
                reinterpret_cast<const float2*>(&A[row + (tile * TILE_SIZE + threadIdx.y) * M])[0];
        }
        
        // Vectorized load for B using float2
        if (col < N && (tile * TILE_SIZE + threadIdx.x) < K) {
            reinterpret_cast<float2*>(&B_shared[threadIdx.x][threadIdx.y])[0] = 
                reinterpret_cast<const float2*>(&B[col * K + tile * TILE_SIZE + threadIdx.x])[0];
        }
        
        __syncthreads();
        
        // Compute with manual loop unrolling
        #pragma unroll 16
        for (int k = 0; k < TILE_SIZE; ++k) {
            if ((tile * TILE_SIZE + k) < K) {
                sum += A_shared[k][threadIdx.x] * B_shared[k][threadIdx.y];
            }
        }
        
        // Single synchronization after computation
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);
    
    auto C = torch::empty({M, N}, A.options());
    
    dim3 threads(16, 16);  // Better occupancy with smaller block size
    dim3 blocks((M + 15) / 16, (N + 15) / 16);
    
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_tiled_kernel", ([&] {
        matmul_transpose_tiled_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Tiled matrix multiplication with transposed matrices (CUDA)");
}
