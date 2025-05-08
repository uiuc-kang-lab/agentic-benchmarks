#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// Define tile dimensions for the output matrix and the k-dimension partitioning
#define TILE_DIM 16
#define K_TILE 16

// Kernel: Each block computes a partial dot product (over a segment of the k dimension) for a tile of C.
// The valid k indices for an upper triangular element C[row, col] are in [row, col+1).
// Blocks are organized in a 3D grid: x and y for tiling C, and z for partitioning the k summation.
// Each block computes a partial sum and uses one atomicAdd per output element to safely accumulate results.

__global__ void upper_triangular_matmul_atomic_kernel(const float* __restrict__ A,
                                                        const float* __restrict__ B,
                                                        float* __restrict__ C,
                                                        int N) {
    // Compute the output element indices based on block and thread indices
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    // Process only valid upper triangular elements
    if (row < N && col < N && row <= col) {
        float sum = 0.0f;
        
        // Determine the k range this block is responsible for along the k-dimension
        int k_block_start = blockIdx.z * K_TILE;
        int k_block_end = (blockIdx.z + 1) * K_TILE;
        if (k_block_end > N) k_block_end = N;

        // For the element C[row, col], the valid k indices are from row to col (inclusive).
        // Adjust the block's k range to the valid portion: [max(k_block_start, row), min(k_block_end, col+1))
        int k_start = (row > k_block_start ? row : k_block_start);
        int k_end = ((col + 1) < k_block_end ? (col + 1) : k_block_end);

        // Accumulate partial sum if there is any overlap in the k-range
        if (k_start < k_end) {
            for (int k = k_start; k < k_end; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            // Use atomicAdd to safely accumulate the partial sum into the final result in global memory
            atomicAdd(&C[row * N + col], sum);
        }
    }
}

// The host function: sets up the grid dimensions and launches the kernel
// Note: It expects that tensors A and B are square and represent an upper triangular multiplication
// where for each C[i,j] (i <= j), the dot-product is taken over k in [i, j+1).

torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    // Initialize C as zeros; this is important because we accumulate partial sums via atomicAdd
    auto C = torch::zeros_like(A);
    
    // Define the block dimensions: each block computes a TILE_DIM x TILE_DIM tile of C
    dim3 block(TILE_DIM, TILE_DIM);
    // Grid dimensions: x and y cover the output matrix, while z partitions the k-dimension
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM,
              (N + TILE_DIM - 1) / TILE_DIM,
              (N + K_TILE - 1) / K_TILE);
    
    // Launch the kernel
    upper_triangular_matmul_atomic_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Atomic cooperative upper triangular matrix multiplication");
}
