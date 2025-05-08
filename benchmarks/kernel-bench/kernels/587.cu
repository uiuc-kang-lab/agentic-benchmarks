#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Use a block size that matches the hardware warp size
#define BLOCK_SIZE 32

// CUDA kernel with improved thread and block indexing for matrix multiplication
template <typename scalar_t>
__global__ void matmul_cuda_kernel_improved(const scalar_t* __restrict__ A,
                                             const scalar_t* __restrict__ B,
                                             scalar_t* __restrict__ C,
                                             int M, int K, int N) {
    // Shared memory tiles for A and B
    __shared__ scalar_t sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ scalar_t sB[BLOCK_SIZE][BLOCK_SIZE];

    // Compute row and column indices using 2D thread indexing
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // index over M
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // index over N
    
    scalar_t sum = 0;
    
    // Loop over tiles in the K dimension
    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        int tiledK = t * BLOCK_SIZE;
        
        // Load tile from matrix A into shared memory
        if (row < M && (tiledK + threadIdx.x) < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + tiledK + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0;

        // Load tile from matrix B into shared memory
        if (col < N && (tiledK + threadIdx.y) < K)
            sB[threadIdx.y][threadIdx.x] = B[(tiledK + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0;

        // Synchronize to ensure the tile is loaded
        __syncthreads();

        // Use unrolling for the inner product computation in the tile
        #pragma unroll
        for (int kIdx = 0; kIdx < BLOCK_SIZE; ++kIdx) {
            sum += sA[threadIdx.y][kIdx] * sB[kIdx][threadIdx.x];
        }

        // Synchronize to prepare for loading the next tile
        __syncthreads();
    }

    // Write the computed value to the output matrix
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Forward function that gets called from Python
torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    // Ensure input tensors are CUDA tensors
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    // Get matrix dimensions
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);

    // Check dimensions compatibility
    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    // Allocate output tensor
    auto C = torch::empty({M, N}, A.options());

    // Define block and grid dimensions based on improved mapping
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the CUDA kernel using the appropriate scalar type
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_cuda_kernel_improved", ([&] {
        matmul_cuda_kernel_improved<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    // Wait for the kernel to finish
    cudaDeviceSynchronize();
    return C;
}

// Pybind11 binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Improved matrix multiplication forward (CUDA)");
}
