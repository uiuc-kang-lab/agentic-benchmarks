#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication with transposed inputs and unrolled loop
template <typename scalar_t>
__global__ void unrolled_matmul_transpose_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    
    // Use shared memory to improve memory access patterns
    __shared__ scalar_t As[16][16];
    __shared__ scalar_t Bs[16][16];
    
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    scalar_t sum = 0;
    
    // Compute matrix multiplication tile by tile
    for (int tile = 0; tile < (K + 15) / 16; ++tile) {
        // Collaborative loading of tiles into shared memory
        if (row < M && (tile * 16 + threadIdx.y) < K) {
            As[threadIdx.x][threadIdx.y] = A[(tile * 16 + threadIdx.y) * M + row];
        } else {
            As[threadIdx.x][threadIdx.y] = 0;
        }
        
        if (col < N && (tile * 16 + threadIdx.x) < K) {
            Bs[threadIdx.y][threadIdx.x] = B[col * K + tile * 16 + threadIdx.x];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }
        
        __syncthreads();
        
        // Compute partial dot product for this tile
        if (row < M && col < N) {
            #pragma unroll
            for (int k = 0; k < 16; ++k) {
                sum += As[threadIdx.x][k] * Bs[threadIdx.y][k];
            }
        }
        
        __syncthreads();
    }
    
    // Write final result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor unrolled_matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    const int BLOCK_SIZE = 16;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(A.type(), "unrolled_matmul_transpose_kernel", ([&] {
        unrolled_matmul_transpose_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &unrolled_matmul_transpose_cuda, "Matrix multiplication with transpose and unrolled loop forward (CUDA)");
}