#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define the tile size for shared memory tiling
#define TILE_SIZE 32

// Hybrid CUDA kernel for lower-triangular matrix multiplication.
// Computes C = tril(A * B) where both A and B are lower triangular.
// Each thread computes one element C[row, col] = sum_{k=col}^{row} A[row, k] * B[k, col].
// The summation is partitioned into blocks of size TILE_SIZE, loaded into shared memory,
// enabling improved data reuse and reduced global memory accesses.

__global__ void efficient_triangular_mm_kernel(const float* __restrict__ A,
                                                const float* __restrict__ B,
                                                float* __restrict__ C,
                                                const int N) {
    // Compute global row and col indices for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    // Only compute for lower triangular region; set upper region to zero
    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;
    
    // Shared memory tiles for A and B
    __shared__ float shA[TILE_SIZE][TILE_SIZE];
    __shared__ float shB[TILE_SIZE][TILE_SIZE];

    // The multiplication for C[row, col] is: sum_{k=col}^{row} A[row, k] * B[k, col].
    // We process this sum in blocks (tiles) of size TILE_SIZE.
    // k0 iterates over the start index of each tile.
    for (int k0 = col; k0 < row + 1; k0 += TILE_SIZE) {
        // Determine the number of valid elements in this tile block
        int tile_len = ((row - k0 + 1) < TILE_SIZE) ? (row - k0 + 1) : TILE_SIZE;

        // Load a tile of A: each thread loads one element from the fixed row "row"
        // A[row, k0 + threadIdx.x] is loaded if (k0 + threadIdx.x) is within [col, row].
        int indexA = k0 + threadIdx.x;
        if (indexA <= row && indexA < N) {
            shA[threadIdx.y][threadIdx.x] = A[row * N + indexA];
        } else {
            shA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load a tile of B: each thread loads one element from column "col"
        // B[k0 + threadIdx.y, col] is valid since k0 >= col, so (k0 + threadIdx.y) >= col.
        int indexB = k0 + threadIdx.y;
        if (indexB < N) {
            shB[threadIdx.y][threadIdx.x] = B[indexB * N + col];
        } else {
            shB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Multiply the loaded tiles. Use unrolling when the tile is full
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            // Only iterate up to the valid tile length
            if (i < tile_len) {
                sum += shA[threadIdx.y][i] * shB[i][threadIdx.x];
            }
        }

        __syncthreads();
    }

    C[row * N + col] = sum;
}

// C++ interface exposed to PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "A and B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    // Define block and grid dimensions using TILE_SIZE
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    efficient_triangular_mm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Efficient triangular matrix multiplication with shared memory tiling");
}
