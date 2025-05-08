#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void matmul_transpose_shared_optimized_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    
    const int TILE_SIZE = 32;
    __shared__ scalar_t A_shared[TILE_SIZE][TILE_SIZE + 1];
    __shared__ scalar_t B_shared[TILE_SIZE][TILE_SIZE + 1];
    
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Early exit for invalid output elements
    if (row >= M || col >= N) return;
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    scalar_t sum = 0;
    
    const int num_full_tiles = K / TILE_SIZE;
    const int remainder = K % TILE_SIZE;

    // Process complete tiles
    for (int tile = 0; tile < num_full_tiles; ++tile) {
        const int a_k = tile * TILE_SIZE + ty;
        A_shared[ty][tx] = A[row + a_k * M];
        
        const int b_k = tile * TILE_SIZE + tx;
        B_shared[tx][ty] = B[col * K + b_k];
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += A_shared[k][tx] * B_shared[k][ty];
        }
        
        __syncthreads();
    }

    // Process remaining elements
    if (remainder > 0) {
        const int a_k = num_full_tiles * TILE_SIZE + ty;
        A_shared[ty][tx] = (a_k < K) ? A[row + a_k * M] : 0;
        
        const int b_k = num_full_tiles * TILE_SIZE + tx;
        B_shared[tx][ty] = (b_k < K) ? B[col * K + b_k] : 0;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < remainder; ++k) {
            sum += A_shared[k][tx] * B_shared[k][ty];
        }
        
        __syncthreads();
    }
    
    C[row * N + col] = sum;
}

torch::Tensor matmul_transpose_cuda_optimized(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);
    
    auto C = torch::empty({M, N}, A.options());
    
    const int BLOCK_SIZE = 32;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE
    );
    
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_shared_optimized_kernel", ([&] {
        matmul_transpose_shared_optimized_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda_optimized, "Optimized transposed matrix multiplication with shared memory");
}