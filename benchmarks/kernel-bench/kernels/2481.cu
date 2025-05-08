#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int TILE_SIZE=32>
__global__ void matmul_transpose_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    
    __shared__ scalar_t tileA[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t tileB[TILE_SIZE][TILE_SIZE];
    
    const int row = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int col = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    scalar_t sum = 0;
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; ++t) {
        const int k_offset = t * TILE_SIZE;
        
        if (k_offset + threadIdx.y < K && row < M) {
            tileA[threadIdx.y][threadIdx.x] = A[(k_offset + threadIdx.y) * M + row];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        if (k_offset + threadIdx.x < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[col * K + k_offset + threadIdx.x];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum = fma(tileA[k][threadIdx.x], tileB[threadIdx.y][k], sum);
        }
        
        if (t < numTiles - 1) {
            __syncthreads();
        }
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
    
    constexpr int TILE_SIZE = 32;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((M + TILE_SIZE - 1) / TILE_SIZE,
                (N + TILE_SIZE - 1) / TILE_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_kernel", ([&] {
        matmul_transpose_kernel<scalar_t, TILE_SIZE><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Optimized matrix multiplication with transpose (CUDA)");
}