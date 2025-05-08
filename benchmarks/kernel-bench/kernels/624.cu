#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH_32 32

// CUDA kernel for matrix multiplication using shared memory tiling
template <typename scalar_t>
__global__ void matmul_cuda_kernel_tuned(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B,
                                   scalar_t* __restrict__ C, int M, int K, int N) {
    __shared__ scalar_t sA[TILE_WIDTH_32][TILE_WIDTH_32];
    __shared__ scalar_t sB[TILE_WIDTH_32][TILE_WIDTH_32];

    int row = blockIdx.y * TILE_WIDTH_32 + threadIdx.y; // M dimension
    int col = blockIdx.x * TILE_WIDTH_32 + threadIdx.x; // N dimension

    scalar_t value = 0;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_WIDTH_32 - 1) / TILE_WIDTH_32; ++t) {
        // Load elements into shared memory
        if (row < M && t * TILE_WIDTH_32 + threadIdx.x < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_WIDTH_32 + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0;

        if (col < N && t * TILE_WIDTH_32 + threadIdx.y < K)
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH_32 + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH_32; ++i) {
            value += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write to output
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

// Forward function
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

    // Define block and grid dimensions
    dim3 threads_per_block(TILE_WIDTH_32, TILE_WIDTH_32);
    dim3 num_blocks((N + TILE_WIDTH_32 - 1) / TILE_WIDTH_32, (M + TILE_WIDTH_32 - 1) / TILE_WIDTH_32);

    // Launch the CUDA kernel
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_cuda_kernel_tuned", ([&] {
        matmul_cuda_kernel_tuned<scalar_t><<<num_blocks, threads_per_block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    // Wait for all kernels to finish
    cudaDeviceSynchronize();

    return C;
}

// Binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Matrix multiplication forward (CUDA)");
}