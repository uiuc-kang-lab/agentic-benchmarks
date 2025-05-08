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
    
    constexpr int TILE_SIZE = 16;
    
    // Shared memory tiles
    __shared__ scalar_t As[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t Bs[TILE_SIZE][TILE_SIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.x * TILE_SIZE + tx;
    const int col = blockIdx.y * TILE_SIZE + ty;
    
    // Initialize accumulator
    scalar_t sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tiles from global memory to shared memory
        const int tile_idx = tile * TILE_SIZE;
        
        if (row < M && (tile_idx + ty) < K) {
            // Load A - note the transposed access pattern
            As[tx][ty] = A[(tile_idx + ty) * M + row];
        } else {
            As[tx][ty] = 0.0f;
        }
        
        if (col < N && (tile_idx + tx) < K) {
            // Load B - note the transposed access pattern
            Bs[tx][ty] = B[col * K + (tile_idx + tx)];
        } else {
            Bs[tx][ty] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum = __fmaf_rn(As[tx][k], Bs[k][ty], sum);
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);
    
    auto C = torch::empty({M, N}, A.options());
    
    constexpr int TILE_SIZE = 16;
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
    m.def("forward", &matmul_transpose_cuda, "Matrix multiplication with transpose (CUDA)");
}