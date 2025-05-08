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
    
    // Use constant memory for frequently accessed values
    const int row = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int col = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Declare shared memory with padding to avoid bank conflicts
    __shared__ scalar_t tileA[TILE_SIZE][TILE_SIZE + 1];
    __shared__ scalar_t tileB[TILE_SIZE][TILE_SIZE + 1];
    
    // Register for accumulation to reduce memory traffic
    scalar_t sum = 0;
    
    // Calculate number of tiles once
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    #pragma unroll 1
    for (int t = 0; t < numTiles; ++t) {
        const int k_offset = t * TILE_SIZE;
        
        // Coalesced memory access pattern
        if (k_offset + threadIdx.y < K && row < M) {
            tileA[threadIdx.y][threadIdx.x] = A[(k_offset + threadIdx.y) * M + row];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0;
        }
        
        if (k_offset + threadIdx.x < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[col * K + k_offset + threadIdx.x];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0;
        }
        
        __syncthreads();
        
        // Manually unrolled inner loop for better instruction-level parallelism
        #pragma unroll 8
        for (int k = 0; k < TILE_SIZE; k += 8) {
            sum += tileA[k][threadIdx.x] * tileB[threadIdx.y][k];
            sum += tileA[k+1][threadIdx.x] * tileB[threadIdx.y][k+1];
            sum += tileA[k+2][threadIdx.x] * tileB[threadIdx.y][k+2];
            sum += tileA[k+3][threadIdx.x] * tileB[threadIdx.y][k+3];
            sum += tileA[k+4][threadIdx.x] * tileB[threadIdx.y][k+4];
            sum += tileA[k+5][threadIdx.x] * tileB[threadIdx.y][k+5];
            sum += tileA[k+6][threadIdx.x] * tileB[threadIdx.y][k+6];
            sum += tileA[k+7][threadIdx.x] * tileB[threadIdx.y][k+7];
        }
        
        // Only sync if not the last iteration
        if (t < numTiles - 1) {
            __syncthreads();
        }
    }
    
    // Coalesced write to global memory
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
    m.def("forward", &matmul_transpose_cuda, "Optimized matrix multiplication with transpose (CUDA)");
}