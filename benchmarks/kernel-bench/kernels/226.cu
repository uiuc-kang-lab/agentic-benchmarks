#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// Device function to load a tile from matrix A into shared memory
__device__ inline void load_tile_A(const float* __restrict__ A, float As[TILE_SIZE][TILE_SIZE], int batch, int M, int K, int t) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = t * TILE_SIZE + threadIdx.x;
    if (row < M && col < K) {
        As[threadIdx.y][threadIdx.x] = A[batch * M * K + row * K + col];
    } else {
        As[threadIdx.y][threadIdx.x] = 0.0f;
    }
}

// Device function to load a tile from matrix B into shared memory
__device__ inline void load_tile_B(const float* __restrict__ B, float Bs[TILE_SIZE][TILE_SIZE], int batch, int K, int N, int t) {
    int row = t * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    if (row < K && col < N) {
        Bs[threadIdx.y][threadIdx.x] = B[batch * K * N + row * N + col];
    } else {
        Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }
}

// CUDA kernel for batched matrix multiplication using shared memory tiling
// with modular device functions
// Computes C = A * B for each batch.
// A: (batch_size, M, K), B: (batch_size, K, N), C: (batch_size, M, N)
__global__ void bmm_tiled_modular_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    int b = blockIdx.z;  // Batch index
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float value = 0.0f;

    // Shared memory tiles for A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        // Load current tiles into shared memory using modular device functions
        load_tile_A(A, As, b, M, K, t);
        load_tile_B(B, Bs, b, K, N, t);
        __syncthreads();

        // Compute partial product for the tile
        for (int i = 0; i < TILE_SIZE; i++) {
            value += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }

    // Write the final computed value to C if within bounds
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
    torch::Tensor C = torch::zeros({batch_size, M, N}, options);

    // Configure grid and block dimensions
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, batch_size);

    bmm_tiled_modular_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Batched matrix multiplication with modular tiling (CUDA)");
}
