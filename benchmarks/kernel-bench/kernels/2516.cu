#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void matmul_transpose_shared_opt_kernel(
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
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    scalar_t sum = 0;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int k_idx = tile * TILE_SIZE;
        
        if (row < M && (k_idx + ty) < K) {
            A_shared[ty][tx] = A[row * K + (k_idx + ty)];
        }
        
        if (col < N && (k_idx + tx) < K) {
            B_shared[tx][ty] = B[col * K + (k_idx + tx)];
        }
        
        __syncthreads();
        
        if (row < M && col < N) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                if ((k_idx + k) < K) {
                    sum += A_shared[k][tx] * B_shared[k][ty];
                }
            }
        }
        
        // Removed unnecessary second __syncthreads()
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
    
    const int BLOCK_SIZE = 32;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_shared_opt_kernel", ([&] {
        matmul_transpose_shared_opt_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Optimized shared memory matmul with reduced syncs");
}
