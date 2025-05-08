#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tuned tile dimension based on experimental block size optimization (e.g., 32x32)
#define TILEDIM 32

// CUDA kernel for matrix multiplication with tuned block size and loop unrolling
// Uses shared memory tiling to reduce global memory accesses

template <typename scalar_t>
__global__ void tuned_matmul_cuda_kernel(const scalar_t* __restrict__ A, 
                                           const scalar_t* __restrict__ B,
                                           scalar_t* __restrict__ C, 
                                           int M, int K, int N) {
    // Allocate shared memory tiles for A and B
    __shared__ scalar_t sA[TILEDIM][TILEDIM];
    __shared__ scalar_t sB[TILEDIM][TILEDIM];

    int row = blockIdx.y * TILEDIM + threadIdx.y; // Row in A and C
    int col = blockIdx.x * TILEDIM + threadIdx.x; // Column in B and C

    scalar_t sum = 0;

    // Loop over tiles in the K dimension
    int numTiles = (K + TILEDIM - 1) / TILEDIM;
    for (int t = 0; t < numTiles; t++) {
        // Load tile from A into shared memory
        if (row < M && t * TILEDIM + threadIdx.x < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + t * TILEDIM + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0;

        // Load tile from B into shared memory
        if (col < N && t * TILEDIM + threadIdx.y < K)
            sB[threadIdx.y][threadIdx.x] = B[(t * TILEDIM + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        // Multiply the two tiles with loop unrolling
        #pragma unroll
        for (int i = 0; i < TILEDIM; i++) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the computed value to the output matrix
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Forward function: verifies dimensions, allocates output and launches the CUDA kernel
torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    // Check that inputs are CUDA tensors
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    // Get matrix dimensions
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);

    // Ensure inner dimensions match
    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    // Allocate output tensor
    auto C = torch::empty({M, N}, A.options());

    // Define block and grid dimensions based on the tuned TILE dimension
    dim3 threads_per_block(TILEDIM, TILEDIM);
    dim3 num_blocks((N + TILEDIM - 1) / TILEDIM, (M + TILEDIM - 1) / TILEDIM);

    // Dispatch CUDA kernel
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "tuned_matmul_cuda_kernel", ([&] {
        tuned_matmul_cuda_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();

    return C;
}

// Binding code for the pybind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Tuned block size matrix multiplication forward (CUDA)");
}
