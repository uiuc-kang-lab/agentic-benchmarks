#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>

// Kernel that computes a tile (stripe) of the upper triangular matrix multiplication.
// It multiplies A (full N x N) with a tile of B (N x tile_width) and writes the result in C (N x tile_width).
// For an output element at (row, tile_col) where global column = col_offset + tile_col, the computation is
// performed only if row <= global column, i.e., for the upper triangular region.
__global__ void pipelined_upper_triangular_matmul_tile_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int col_offset,
    int tile_width
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tile_col = blockIdx.x * blockDim.x + threadIdx.x;
    int col = col_offset + tile_col;
    if (row < N && tile_col < tile_width) {
        float sum = 0.0f;
        if (row <= col) {
            // Only iterate from k = row to k = col in accordance with the upper triangular condition
            for (int k = row; k <= col; k++) {
                sum += A[row * N + k] * B[k * tile_width + tile_col];
            }
        }
        C[row * tile_width + tile_col] = sum;
    }
}

// This function implements the pipelined upper triangular matrix multiplication by partitioning
// the B and C matrices (i.e. the column dimension) into tiles. For each tile, it uses cudaMemcpy2DAsync
// to transfer the tile of B from host to device, launches the computation kernel on a dedicated stream,
// and then asynchronously transfers the computed C tile back to host memory. Two streams and associated
// double buffers are used to overlap data transfers with computation.

torch::Tensor pipelined_upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    // Assume A and B are square matrices stored in contiguous host memory (ideally pinned) of size N x N
    int N = A.size(0);

    // Choose a tile width for partitioning the column dimension; adjust as needed for your GPU
    int tile_width = 256;
    int num_tiles = (N + tile_width - 1) / tile_width;

    // Allocate the output tensor C on host with the same options as A
    auto C = torch::empty({N, N}, A.options());

    // Allocate device memory for the full matrix A and copy it from host to device.
    // A is used in all tile computations so we copy it once.
    float* d_A = nullptr;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMemcpy(d_A, A.data_ptr<float>(), N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Create two CUDA streams for pipelining transfers and kernel execution
    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    // Allocate double buffers: one pair for B tile and one for C tile per stream
    float* d_B_tile[2];
    float* d_C_tile[2];
    for (int i = 0; i < 2; i++) {
        cudaMalloc(&d_B_tile[i], N * tile_width * sizeof(float));
        cudaMalloc(&d_C_tile[i], N * tile_width * sizeof(float));
    }

    dim3 blockDim(16, 16);

    // Loop over each tile (stripe) along the column dimension
    for (int t = 0; t < num_tiles; t++) {
        int current_tile_width = std::min(tile_width, N - t * tile_width);
        int col_offset = t * tile_width;
        dim3 gridDim((current_tile_width + blockDim.x - 1) / blockDim.x,
                     (N + blockDim.y - 1) / blockDim.y);
        cudaStream_t stream = streams[t % 2];

        // Asynchronously copy the B tile from host to device.
        // Since B is stored in row-major order (pitch = N * sizeof(float)),
        // every row's tile starting at column 'col_offset' is contiguous for current_tile_width elements.
        cudaMemcpy2DAsync(
            d_B_tile[t % 2],                       // destination pointer in device
            current_tile_width * sizeof(float),      // destination pitch (row length in bytes)
            B.data_ptr<float>() + col_offset,        // source pointer (start at column offset for row 0)
            N * sizeof(float),                       // source pitch (full row length in bytes)
            current_tile_width * sizeof(float),      // width of the submatrix in bytes
            N,                                       // number of rows
            cudaMemcpyHostToDevice, stream);

        // Launch the kernel on the current stream to compute the corresponding C tile
        pipelined_upper_triangular_matmul_tile_kernel<<<gridDim, blockDim, 0, stream>>>(
            d_A, d_B_tile[t % 2], d_C_tile[t % 2],
            N, col_offset, current_tile_width
        );

        // Asynchronously copy the computed C tile from device back to host.
        cudaMemcpy2DAsync(
            C.data_ptr<float>() + col_offset,      // destination pointer in host for column offset
            N * sizeof(float),                       // destination pitch (full row length in bytes)
            d_C_tile[t % 2],                         // source pointer in device (tile buffer)
            current_tile_width * sizeof(float),      // source pitch (tile row length in bytes)
            current_tile_width * sizeof(float),      // width of the tile in bytes
            N,                                       // number of rows
            cudaMemcpyDeviceToHost, stream);
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
    m.def("forward", &pipelined_upper_triangular_matmul, "Pipelined upper triangular matrix multiplication with streams");
}
