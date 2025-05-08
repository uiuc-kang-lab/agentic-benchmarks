#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define ELEMENTS_PER_THREAD 4
#define TILE_SIZE 32

template <typename scalar_t>
__global__ void matmul_transpose_strided_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    
    __shared__ scalar_t As[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t Bs[TILE_SIZE][TILE_SIZE];
    
    const int bx = blockIdx.x * TILE_SIZE;
    const int by = blockIdx.y * TILE_SIZE;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Each thread computes multiple elements
    scalar_t acc[ELEMENTS_PER_THREAD][ELEMENTS_PER_THREAD] = {0};
    
    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tiles with stride pattern
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i += BLOCK_SIZE) {
            for (int j = 0; j < TILE_SIZE; j += BLOCK_SIZE) {
                if ((tile * TILE_SIZE + i + tx) < K && (by + j + ty) < M)
                    As[j + ty][i + tx] = __ldg(&A[(tile * TILE_SIZE + i + tx) * M + (by + j + ty)]);
                if ((bx + j + ty) < N && (tile * TILE_SIZE + i + tx) < K)
                    Bs[j + ty][i + tx] = __ldg(&B[(bx + j + ty) * K + (tile * TILE_SIZE + i + tx)]);
            }
        }
        
        __syncthreads();
        
        // Compute partial products with stride pattern
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                #pragma unroll
                for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
                    int row = ty + i * BLOCK_SIZE;
                    int col = tx + j * BLOCK_SIZE;
                    if (row < TILE_SIZE && col < TILE_SIZE) {
                        acc[i][j] += As[row][k] * Bs[col][k];
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results with stride pattern
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        #pragma unroll
        for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
            int row = by + ty + i * BLOCK_SIZE;
            int col = bx + tx + j * BLOCK_SIZE;
            if (row < M && col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);
    
    auto C = torch::empty({M, N}, A.options());
    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_transpose_strided_kernel", ([&] {
        matmul_transpose_strided_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Matrix multiplication with transpose using strided pattern (CUDA)");
}