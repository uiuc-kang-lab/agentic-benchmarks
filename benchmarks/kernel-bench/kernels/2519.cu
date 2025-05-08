#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void matmul_transpose_shared_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    
    // Use a smaller tile size and single-buffered shared memory
    const int TILE_SIZE = 32;
    __shared__ scalar_t A_shared[TILE_SIZE][TILE_SIZE + 1];
    __shared__ scalar_t B_shared[TILE_SIZE][TILE_SIZE + 1];
    
    // Compute indices for C (dimensions M x N), where M corresponds to rows of C
    int i = blockIdx.x * TILE_SIZE + threadIdx.x; // index along M (from A's second dim)
    int j = blockIdx.y * TILE_SIZE + threadIdx.y; // index along N (from B's first dim)
    
    scalar_t sum = 0;
    
    // Loop over tiles of the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        int pA = t * TILE_SIZE + threadIdx.y; // index in K for A
        if (i < M && pA < K)
            A_shared[threadIdx.y][threadIdx.x] = A[i + pA * M];
        else
            A_shared[threadIdx.y][threadIdx.x] = 0;
        
        int pB = t * TILE_SIZE + threadIdx.x; // index in K for B
        if (j < N && pB < K)
            B_shared[threadIdx.x][threadIdx.y] = B[j * K + pB];
        else
            B_shared[threadIdx.x][threadIdx.y] = 0;
        
        __syncthreads();
        
#pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += A_shared[k][threadIdx.x] * B_shared[k][threadIdx.y];
        }
        
        __syncthreads();
    }
    
    if (i < M && j < N) {
        C[i * N + j] = sum;
    }
}

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);
    
    auto C = torch::empty({M, N}, A.options());
    
    const int BLOCK_SIZE = 64;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_shared_kernel", ([&] {
        matmul_transpose_shared_kernel<scalar_t><<<blocks, threads>>>(
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