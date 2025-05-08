#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication with transposed inputs and optimized memory access
template <typename scalar_t>
__global__ void matmul_transpose_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    
    // Tiled kernel using shared memory for improved memory reuse
    const int BLOCK_SIZE = 32;
    int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int col = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    scalar_t sum = 0;
    
    __shared__ scalar_t sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ scalar_t sB[BLOCK_SIZE][BLOCK_SIZE];

    const int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        // Load a tile of A. A is of shape (K, M), accessed as A[k*M + row]
        int kA = t * BLOCK_SIZE + threadIdx.y;
        if (row < M && kA < K)
            sA[threadIdx.x][threadIdx.y] = __ldg(&A[kA * M + row]);
        else
            sA[threadIdx.x][threadIdx.y] = 0;
            
        // Load a tile of B. B is of shape (N, K), accessed as B[col*K + k]
        int kB = t * BLOCK_SIZE + threadIdx.x;
        if (col < N && kB < K)
            sB[threadIdx.x][threadIdx.y] = __ldg(&B[col * K + kB]);
        else
            sB[threadIdx.x][threadIdx.y] = 0;
            
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += sA[threadIdx.x][k] * sB[k][threadIdx.y];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    // Get dimensions
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);

    // Create output tensor
    auto C = torch::empty({M, N}, A.options());

    // Calculate grid and block dimensions - ensure alignment
    const int BLOCK_SIZE = 32;  // Increased block size for better occupancy
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
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