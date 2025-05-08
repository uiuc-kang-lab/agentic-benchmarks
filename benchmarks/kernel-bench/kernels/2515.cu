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
    
    const int TILE_SIZE = 64;
    __shared__ scalar_t A_shared[2][TILE_SIZE][TILE_SIZE + 1];
    __shared__ scalar_t B_shared[2][TILE_SIZE][TILE_SIZE + 1];
    
    const int row = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int col = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    scalar_t sum = 0;
    scalar_t loading_a[2] = {0, 0};
    scalar_t loading_b[2] = {0, 0};
    int current = 0;

    // Preload first tiles
    if (row < M && (threadIdx.y < TILE_SIZE) && (threadIdx.y < K))
        A_shared[current][threadIdx.y][threadIdx.x] = A[row + (threadIdx.y) * M];
    
    if (col < N && (threadIdx.x < K))
        B_shared[current][threadIdx.x][threadIdx.y] = B[col * K + threadIdx.x];
    
    __syncthreads();

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int next = (current + 1) % 2;

        // Preload next tiles while computing current
        if (t + 1 < (K + TILE_SIZE - 1) / TILE_SIZE) {
            int offset = (t + 1) * TILE_SIZE;
            if (row < M && (offset + threadIdx.y) < K)
                loading_a[next] = A[row + (offset + threadIdx.y) * M];
            
            if (col < N && (offset + threadIdx.x) < K)
                loading_b[next] = B[col * K + offset + threadIdx.x];
        }

        // Compute current tile
#pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += A_shared[current][k][threadIdx.x] * 
                   B_shared[current][k][threadIdx.y];
        }

        // Async load next tiles
        if (t + 1 < (K + TILE_SIZE - 1) / TILE_SIZE) {
            A_shared[next][threadIdx.y][threadIdx.x] = loading_a[next];
            B_shared[next][threadIdx.x][threadIdx.y] = loading_b[next];
        }

        current = next;
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
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