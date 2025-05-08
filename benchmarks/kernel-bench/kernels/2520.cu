#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void matmul_transpose_shared_unrolled_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    
    const int TILE_SIZE = 32;
    __shared__ scalar_t A_shared[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t B_shared[TILE_SIZE][TILE_SIZE];
    
    const int row = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int col = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Use multiple accumulators for better instruction-level parallelism
    scalar_t sum0 = 0;
    scalar_t sum1 = 0;
    scalar_t sum2 = 0;
    scalar_t sum3 = 0;
    
    // Iterate over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Collaborative loading of tiles into shared memory
        if (row < M && (tile * TILE_SIZE + ty) < K) {
            A_shared[ty][tx] = A[(tile * TILE_SIZE + ty) * M + row];
        } else {
            A_shared[ty][tx] = 0;
        }
        
        if (col < N && (tile * TILE_SIZE + tx) < K) {
            B_shared[tx][ty] = B[col * K + tile * TILE_SIZE + tx];
        } else {
            B_shared[tx][ty] = 0;
        }
        
        __syncthreads();
        
        if (row < M && col < N) {
            // Manual unrolling of the computation loop
            #pragma unroll 8
            for (int k = 0; k < TILE_SIZE; k += 4) {
                if ((tile * TILE_SIZE + k) < K) {
                    sum0 += A_shared[k][tx] * B_shared[k][ty];
                    sum1 += A_shared[k+1][tx] * B_shared[k+1][ty];
                    sum2 += A_shared[k+2][tx] * B_shared[k+2][ty];
                    sum3 += A_shared[k+3][tx] * B_shared[k+3][ty];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Combine partial sums and write final result
    if (row < M && col < N) {
        C[row * N + col] = sum0 + sum1 + sum2 + sum3;
    }
}

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);
    
    auto C = torch::empty({M, N}, A.options());
    
    const int BLOCK_SIZE = 32;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_shared_unrolled_kernel", ([&] {
        matmul_transpose_shared_unrolled_kernel<scalar_t><<<blocks, threads>>>(
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