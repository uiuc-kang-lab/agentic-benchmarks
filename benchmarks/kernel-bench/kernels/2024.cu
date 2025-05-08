#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define the tile size to partition both the output and the k-dimension
#define TILE_SIZE 32

// This kernel partitions the reduction (k) dimension into tiles. Each thread computes a partial sum
// over a k-tile and uses an atomicAdd to accumulate the results into global memory. Atomic operations
// are used only to combine partial results from different k-tiles, minimizing contention.
__global__ void atomic_tiled_triangular_mm_kernel(const float* __restrict__ A,
                                                   const float* __restrict__ B,
                                                   float* __restrict__ C,
                                                   int N) {
    // Determine global output indices
    int out_row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int out_col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Each block in the z-dimension handles a tile from the k-dimension
    int k_tile_start = blockIdx.z * TILE_SIZE;
    int k_tile_end = k_tile_start + TILE_SIZE;
    if (k_tile_end > N) k_tile_end = N;

    // Out-of-bound or upper triangular regions (for lower triangular matrices) are skipped
    if (out_row >= N || out_col >= N) return;
    if (out_row < out_col) return;

    float partial = 0.0f;

    // Load a tile of A and B from global memory into shared memory to improve coalescing
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    // Each thread loads one element from A for the current k-tile.
    int k_index_A = k_tile_start + threadIdx.x; // column index in A
    if (k_index_A < N && k_index_A <= out_row) {
        tile_A[threadIdx.y][threadIdx.x] = A[out_row * N + k_index_A];
    } else {
        tile_A[threadIdx.y][threadIdx.x] = 0.0f;
    }

    // Each thread loads one element from B for the current k-tile.
    int k_index_B = k_tile_start + threadIdx.y; // row index in B
    if (k_index_B < N && k_index_B >= out_col) {
        tile_B[threadIdx.y][threadIdx.x] = B[k_index_B * N + out_col];
    } else {
        tile_B[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Determine the valid range within this k-tile that contributes to C[out_row, out_col].
    int valid_start = (out_col > k_tile_start) ? out_col : k_tile_start;
    int valid_end = (k_tile_end < (out_row + 1)) ? k_tile_end : (out_row + 1);

    // Translate the valid range into local indices in the shared memory tile
    int local_start = valid_start - k_tile_start;
    int local_end = valid_end - k_tile_start;

    // Accumulate products over the valid local indices
    for (int s = local_start; s < local_end; ++s) {
        partial += tile_A[threadIdx.y][s] * tile_B[s][threadIdx.x];
    }

    // Use an atomic operation to add the partial sum to the global result
    atomicAdd(&C[out_row * N + out_col], partial);
}

// Host function: verifies inputs, initializes output to zero, and launches the kernel across a 3D grid
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    // Initialize output C to zero, as atomic adds will accumulate partial sums
    auto C = torch::zeros_like(A);

    // Configure the grid dimensions: grid.x and grid.y partition the output matrix,
    // while grid.z partitions the k-dimension (reduction dimension) into tiles
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (N + TILE_SIZE - 1) / TILE_SIZE,
              (N + TILE_SIZE - 1) / TILE_SIZE);

    atomic_tiled_triangular_mm_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Atomic tiled lower triangular matrix multiplication (CUDA)");
}
