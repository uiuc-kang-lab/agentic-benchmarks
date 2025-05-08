#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int BLOCK_SIZE>
__global__ void matmul_transpose_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    
    __shared__ scalar_t As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ scalar_t Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    scalar_t sum = 0;
    
    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        // Load tiles into shared memory
        if (tile * BLOCK_SIZE + ty < K && row < M)
            As[tx][ty] = A[(tile * BLOCK_SIZE + ty) * M + row];
        else
            As[tx][ty] = 0;
            
        if (tile * BLOCK_SIZE + tx < K && col < N)
            Bs[ty][tx] = B[col * K + tile * BLOCK_SIZE + tx];
        else
            Bs[ty][tx] = 0;
            
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[tx][k] * Bs[ty][k];
        }
        
        __syncthreads();
    }
    
    // Warp-level reduction within each warp
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Write result
    if (row < M && col < N) {
        if (threadIdx.x % warpSize == 0) {
            C[row * N + col] = sum;
        }
    }
}

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);
    
    auto C = torch::empty({M, N}, A.options());
    
    constexpr int BLOCK_SIZE = 32;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_kernel", ([&] {
        matmul_transpose_kernel<scalar_t, BLOCK_SIZE><<<blocks, threads>>>(
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