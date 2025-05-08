#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile size for shared memory tiling
#define TILE_SIZE 16

// CUDA kernel for batched matrix multiplication using shared memory tiling
// Computes C = A * B for each batch.
// A: (batch_size, M, K), B: (batch_size, K, N), C: (batch_size, M, N)
__global__ void bmm_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    // Batch index from grid z dimension
    int b = blockIdx.z;

    // Compute row and column indices in the C matrix
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float value = 0.0f;
    
    // Declare shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Loop over tiles of the input matrices
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        // Load a tile of A into shared memory
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (row < M && aCol < K) {
            As[threadIdx.y][threadIdx.x] = A[b * M * K + row * K + aCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load a tile of B into shared memory
        int bRow = t * TILE_SIZE + threadIdx.y;
        if (col < N && bRow < K) {
            Bs[threadIdx.y][threadIdx.x] = B[b * K * N + bRow * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Multiply the two tiles together
        for (int k = 0; k < TILE_SIZE; k++) {
            value += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result in C if within bounds
    if (row < M && col < N) {
        C[b * M * N + row * N + col] = value;
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

    // Configure block and grid dimensions
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, batch_size);

    bmm_tiled_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Batched matrix multiplication with tiling (CUDA)");
}
