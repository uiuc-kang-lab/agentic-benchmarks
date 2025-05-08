#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Define tile sizes
#define TILE_WIDTH 32   // Tile width for K dimension
#define TILE_L 16       // Tile size for L (reduction) dimension
#define BLOCK_Y 4       // Number of (b,i,j) outputs computed per block

// We use a 2D grid: 
//   grid.x covers the (BATCH_chunk * I * J) elements processed per stream (with each block handling BLOCK_Y of them)
//   grid.y covers tiles of the K dimension
// Each thread block is configured as: blockDim.x = TILE_WIDTH (for k tiling) and blockDim.y = BLOCK_Y

__global__ void einsum_kernel_streamed_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH_chunk, int I, int J, int L, int K
) {
    // Compute the linear index for the (b,i,j) combination within the batch chunk
    int out_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (out_idx >= BATCH_chunk * I * J) return;

    // Map out_idx to (b, i, j). Note: A and C pointers are assumed to start from the beginning of the batch chunk
    int b = out_idx / (I * J);
    int rem = out_idx % (I * J);
    int i = rem / J;
    int j = rem % J;

    // Compute the k coordinate based on grid.y and threadIdx.x
    int k_base = blockIdx.y * TILE_WIDTH;
    int k = k_base + threadIdx.x;

    float sum = 0.0f;

    // Shared memory to hold a tile from B, working on the current block's k tile and a tile of L values
    __shared__ float B_tile[TILE_L][TILE_WIDTH];

    // Loop over L dimension in tiles of size TILE_L
    for (int l_tile = 0; l_tile < L; l_tile += TILE_L) {
         int current_tile = min(TILE_L, L - l_tile);

         // Load a tile of B from global memory into shared memory
         // Only one row of the block (e.g., threadIdx.y == 0) performs the load
         if (threadIdx.y == 0) {
             for (int r = 0; r < current_tile; r++) {
                 int global_l = l_tile + r;
                 int global_k = k_base + threadIdx.x;
                 if (global_k < K) {
                      B_tile[r][threadIdx.x] = B[global_l * K + global_k];
                 } else {
                      B_tile[r][threadIdx.x] = 0.0f;
                 }
             }
         }
         __syncthreads();

         // Use the loaded tile to update the sum
         // Each thread loads its corresponding A element (for fixed b,i,j) for the current sub-tile of L
         for (int r = 0; r < current_tile; r++) {
             // Access A: shape is (BATCH_chunk, I, J, L)
             float a_val = A[b * (I * J * L) + i * (J * L) + j * L + (l_tile + r)];
             if (k < K) {
                 sum += a_val * B_tile[r][threadIdx.x];
             }
         }
         __syncthreads();
    }

    // Write output back if k index is valid
    if (k < K) {
         C[b * (I * J * K) + i * (J * K) + j * K + k] = sum;
    }
}

// Forward function launches the kernel in streams. It partitions the batch dimension to overlap kernel execution.
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch: A.size(3) must equal B.size(0)");

    int BATCH = A.size(0);
    int I = A.size(1);
    int J = A.size(2);
    int L = A.size(3);
    int K = B.size(1);

    auto C = torch::zeros({BATCH, I, J, K}, A.options());

    // Use multiple streams to overlap kernel execution for different batch chunks
    int num_streams = 2;
    int batch_chunk = (BATCH + num_streams - 1) / num_streams;
    std::vector<cudaStream_t> streams(num_streams);

    for (int s = 0; s < num_streams; s++) {
        cudaStreamCreate(&streams[s]);
    }

    // Kernel configuration
    dim3 block(TILE_WIDTH, BLOCK_Y);
    int grid_y = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    // Launch a kernel for each stream with its corresponding batch chunk
    for (int s = 0; s < num_streams; s++) {
        int start_batch = s * batch_chunk;
        int current_batch = std::min(batch_chunk, BATCH - start_batch);
        if (current_batch <= 0) break;

        const float* A_ptr = A.data_ptr<float>() + start_batch * (I * J * L);
        float* C_ptr = C.data_ptr<float>() + start_batch * (I * J * K);

        int total_out = current_batch * I * J; // total (b,i,j) outputs in this chunk
        int grid_x = (total_out + BLOCK_Y - 1) / BLOCK_Y;
        dim3 grid(grid_x, grid_y);

        einsum_kernel_streamed_tiled<<<grid, block, 0, streams[s]>>>(
            A_ptr, B.data_ptr<float>(), C_ptr,
            current_batch, I, J, L, K
        );
    }

    // Synchronize and destroy streams
    for (int s = 0; s < num_streams; s++) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication with streamed batch tiling and shared memory (CUDA)");
}
