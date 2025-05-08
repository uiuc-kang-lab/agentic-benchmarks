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
    
    const int TILE_SIZE = 16;
    __shared__ scalar_t A_shared[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t B_shared[TILE_SIZE][TILE_SIZE];
    
    const int row = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int col = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    scalar_t sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load data into shared memory
        if (threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE) {
            int k = tile * TILE_SIZE + threadIdx.y;
            if (k < K && row < M) {
                A_shared[threadIdx.y][threadIdx.x] = A[k * M + row];
            } else {
                A_shared[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            k = tile * TILE_SIZE + threadIdx.x;
            if (k < K && col < N) {
                B_shared[threadIdx.x][threadIdx.y] = B[col * K + k];
            } else {
                B_shared[threadIdx.x][threadIdx.y] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        if (row < M && col < N) {
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += A_shared[k][threadIdx.x] * B_shared[k][threadIdx.y];
            }
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
    
    const int TILE_SIZE = 16;
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
    m.def("forward", &matmul_transpose_cuda, "Matrix multiplication with transpose forward (CUDA)");
}