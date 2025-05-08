#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define TILE_SIZE to experiment with block sizes. 
// Set TILE_SIZE to 32 for 1024 threads per block, which is a configuration worth testing on NVIDIA H100.
#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

// Kernel using shared memory tiling and loop unrolling. The tile size can be tuned to experiment with various block sizes
__global__ void tiled_bmm_kernel_opt(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    // Shared memory tiles for A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Determine batch (bz), row, and column indices
    int batch = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;

    // Pointers for the current batch
    const float* a_batch = A + batch * M * K;
    const float* b_batch = B + batch * K * N;

    // Loop over tiles in the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        int A_col = t * TILE_SIZE + threadIdx.x;
        int B_row = t * TILE_SIZE + threadIdx.y;

        // Load data into shared memory for A
        if (row < M && A_col < K) {
            As[threadIdx.y][threadIdx.x] = a_batch[row * K + A_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load data into shared memory for B
        if (B_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = b_batch[B_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();
        
        // Compute partial dot product for this tile
        // Using unrolling for the inner loop
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            sum = As[threadIdx.y][i] * Bs[i][threadIdx.x] + sum;
        }

        __syncthreads();
    }

    // Write the result
    if (row < M && col < N) {
        C[batch * M * N + row * N + col] = sum;
    }
}

// Forward function to launch the kernel
torch::Tensor forward_bmm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions (K) must match");

    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({batch_size, M, N}, options);

    // Set block dimensions based on TILE_SIZE. Here we experiment with a block size of TILE_SIZE x TILE_SIZE threads
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE,
        batch_size
    );

    // Launch kernel
    tiled_bmm_kernel_opt<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Tiled batched matrix multiplication with optimized block size (CUDA)");
}
