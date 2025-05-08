#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void matmul_transpose_vectorized_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {

    const int TILE_SIZE = 32;
    const int VECTOR_SIZE = 4;
    __shared__ scalar_t A_shared[TILE_SIZE][TILE_SIZE + 1];
    __shared__ scalar_t B_shared[TILE_SIZE][TILE_SIZE + 1];

    const int row = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int col = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    scalar_t sum = 0;
    
    for (int tile_k = 0; tile_k < K; tile_k += TILE_SIZE) {
        // Load A tile with vectorized access
        if (row < M) {
            #pragma unroll
            for (int v = 0; v < VECTOR_SIZE; ++v) {
                int k_idx = tile_k + threadIdx.y * VECTOR_SIZE + v;
                A_shared[threadIdx.y * VECTOR_SIZE + v][threadIdx.x] = 
                    (k_idx < K) ? __ldg(&A[k_idx * M + row]) : 0;
            }
        }

        // Load B tile with vectorized access
        if (col < N) {
            #pragma unroll
            for (int v = 0; v < VECTOR_SIZE; ++v) {
                int k_idx = tile_k + threadIdx.x * VECTOR_SIZE + v;
                B_shared[threadIdx.x * VECTOR_SIZE + v][threadIdx.y] = 
                    (k_idx < K) ? __ldg(&B[col * K + k_idx]) : 0;
            }
        }

        __syncthreads();

        // Compute partial sum with double buffer
        if (row < M && col < N) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE*VECTOR_SIZE; ++k) {
                if (tile_k + k < K) {
                    sum += A_shared[k][threadIdx.x] * B_shared[k][threadIdx.y];
                }
            }
        }
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
    
    const int BLOCK_SIZE = 32;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_vectorized_kernel", ([&] {
        matmul_transpose_vectorized_kernel<scalar_t><<<blocks, threads>>>(
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
