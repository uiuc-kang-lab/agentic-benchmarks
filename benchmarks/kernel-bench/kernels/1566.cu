/*
Combined CUDA extension: pipelined_balanced
This implementation merges pipelined data transfers with a balanced workload kernel across tiles.
It partitions the B (and C) matrices into column tiles and overlaps device-host transfers with kernel computation
using double buffering and two CUDA streams. Within each tile, a balanced kernel uses a strided loop
assignment to distribute the work of computing only the upper triangular elements across the threads.

Compile with torch extensions.
*/

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>

// Kernel: Compute a tile (stripe) of the upper triangular matrix multiplication using balanced workload.
// A is a full N x N matrix. B_tile is a tile of B corresponding to columns [col_offset, col_offset+tile_width).
// C_tile is the output tile of C. Each thread processes multiple output columns with a stride across the tile.
// Only elements with row <= global column (col_offset + tile_col) are computed; others remain zero.
__global__ void pipelined_balanced_upper_triangular_tile_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B_tile,
    float* __restrict__ C_tile,
    int N,
    int col_offset,
    int tile_width
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tile_col_start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Loop over tile columns in steps to balance the workload among threads
    for (int tile_col = tile_col_start; tile_col < tile_width; tile_col += stride) {
        int col = col_offset + tile_col; // global column index
        float sum = 0.0f;
        // Only compute if row is valid and in the upper-triangular region
        if (row < N && row <= col) {
            // Accumulate the result: summing from k = row to k = col
            for (int k = row; k <= col; ++k) {
                float a_val = __ldg(&A[row * N + k]);
                float b_val = __ldg(&B_tile[k * tile_width + tile_col]);
                sum += a_val * b_val;
            }
            C_tile[row * tile_width + tile_col] = sum;
        } else if (row < N) {
            // Ensuring the lower triangular region is zero
            C_tile[row * tile_width + tile_col] = 0.0f;
        }
    }
}

// The host function implements pipelined upper triangular matrix multiplication by
// partitioning the B and C matrices (i.e., the column dimension) into tiles.
// It employs double buffering and multiple CUDA streams to overlap data transfers with computation.
// Within each tile, a balanced workload kernel is launched.

torch::Tensor pipelined_balanced_upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    // Assume A and B are square matrices in contiguous (preferably pinned) host memory
    int N = A.size(0);

    // Choose a tile width for partitioning the column dimension. Adjust as needed.
    int tile_width = 256;
    int num_tiles = (N + tile_width - 1) / tile_width;

    // Allocate the output tensor C on host with the same options as A
    auto C = torch::empty({N, N}, A.options());

    // Allocate device memory for the full matrix A and copy from host to device (only once).
    float* d_A = nullptr;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMemcpy(d_A, A.data_ptr<float>(), N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Create two CUDA streams for pipelining transfers and kernel execution
    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    // Allocate double buffers for B tile and C tile per stream
    float* d_B_tile[2];
    float* d_C_tile[2];
    for (int i = 0; i < 2; i++) {
        cudaMalloc(&d_B_tile[i], N * tile_width * sizeof(float));
        cudaMalloc(&d_C_tile[i], N * tile_width * sizeof(float));
    }

    // Define block dimensions; these can be tuned for the target GPU
    dim3 blockDim(16, 16);

    // Loop over each tile (stripe) along the column dimension
    for (int t = 0; t < num_tiles; t++) {
        int current_tile_width = std::min(tile_width, N - t * tile_width);
        int col_offset = t * tile_width;
        // Grid dimension: cover all rows and current tile columns
        dim3 gridDim((current_tile_width + blockDim.x - 1) / blockDim.x,
                     (N + blockDim.y - 1) / blockDim.y);
        cudaStream_t stream = streams[t % 2];

        // Asynchronously copy the B tile from host to device.
        // B is stored in row-major order, so each row's submatrix starting at column 'col_offset'
        cudaMemcpy2DAsync(
            d_B_tile[t % 2],                     // destination pointer in device tile buffer
            current_tile_width * sizeof(float),    // destination pitch (row length in bytes for the tile)
            B.data_ptr<float>() + col_offset,      // source pointer starting at column offset
            N * sizeof(float),                     // source pitch (row length in full B matrix)
            current_tile_width * sizeof(float),    // width of the submatrix in bytes
            N,                                     // number of rows
            cudaMemcpyHostToDevice,
            stream
        );

        // Launch the balanced workload kernel on the current stream to compute this C tile
        pipelined_balanced_upper_triangular_tile_kernel<<<gridDim, blockDim, 0, stream>>>(
            d_A, d_B_tile[t % 2], d_C_tile[t % 2],
            N, col_offset, current_tile_width
        );

        // Asynchronously copy the computed C tile from device back to host.
        cudaMemcpy2DAsync(
            C.data_ptr<float>() + col_offset,    // destination pointer in host for this tile
            N * sizeof(float),                     // destination pitch (full row length in bytes of C)
            d_C_tile[t % 2],                       // source pointer in device tile buffer
            current_tile_width * sizeof(float),    // source pitch (tile row length in bytes)
            current_tile_width * sizeof(float),    // width of the tile in bytes
            N,                                     // number of rows
            cudaMemcpyDeviceToHost,
            stream
        );
    }

    // Synchronize both streams to ensure all asynchronous operations are complete
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    // Free device memory and destroy the streams
    for (int i = 0; i < 2; i++) {
        cudaFree(d_B_tile[i]);
        cudaFree(d_C_tile[i]);
    }
    cudaFree(d_A);
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &pipelined_balanced_upper_triangular_matmul, 
          "Pipelined and balanced upper triangular matrix multiplication combining streams and balanced workload distribution");
}
