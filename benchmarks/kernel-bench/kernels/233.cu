#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Use an increased tile size to balance occupancy and shared memory usage
#define TILE_SIZE 32

// Optimized CUDA kernel for batched matrix multiplication
// Computes C[b, m, n] = sum_{k} A[b, m, k] * B[b, k, n]
// A: (batch_size, M, K), B: (batch_size, K, N), C: (batch_size, M, N)
__global__ void bmm_optimized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    // Obtain batch index from the grid's z-dimension
    int b = blockIdx.z;
    // Compute the row and column indices for C
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Declare shared memory tiles for A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Accumulator for the dot product
    float sum = 0.0f;

    // Precompute the base pointers for current batch to reduce arithmetic overhead
    const float* A_batch = A + b * M * K;
    const float* B_batch = B + b * K * N;

    // Number of tiles along the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        int tiledCol = t * TILE_SIZE + threadIdx.x;
        int tiledRow = t * TILE_SIZE + threadIdx.y;

        // Load a tile from A into shared memory with boundary check
        if (row < M && tiledCol < K) {
            As[threadIdx.y][threadIdx.x] = A_batch[row * K + tiledCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load a tile from B into shared memory with boundary check
        if (tiledRow < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B_batch[tiledRow * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize to ensure the tile is fully loaded
        __syncthreads();

        // Compute the partial product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Synchronize to ensure all threads are done before loading new tiles
        __syncthreads();
    }

    // Write the result to global memory if within bounds
    if (row < M && col < N) {
        C[b * M * N + row * N + col] = sum;
    }
}

// Host function to launch the optimized kernel
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

    // Configure CUDA block and grid dimensions
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, batch_size);

    // Launch the optimized kernel
    bmm_optimized_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Optimized batched matrix multiplication (CUDA)");
}
