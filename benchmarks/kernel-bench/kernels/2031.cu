#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Templated kernel that uses shared memory tiling for lower triangular matrix multiplication.
// The block (tile) size is a template parameter, allowing exploration of different configurations (8, 16, 32).

template <int BLOCK_SIZE>
__global__ void triangular_mm_tiled_template_kernel(const float* __restrict__ A,
                                                     const float* __restrict__ B,
                                                     float* __restrict__ C,
                                                     int N) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE+1]; // Padding to avoid bank conflicts
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE+1];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row >= N || col >= N) return;

    // For a lower triangular matrix, only compute when row >= col
    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;
    int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Loop over k dimension in tiles
    for (int t = 0; t < numTiles; t++) {
        int tile_start = t * BLOCK_SIZE;
        if (tile_start > row) break;  // No contribution if the tile start exceeds current row index

        // Load A tile element: row is fixed; columns vary in the tile
        int a_col = tile_start + threadIdx.x;
        if (a_col < N)
            As[threadIdx.y][threadIdx.x] = __ldg(&A[row * N + a_col]);
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B tile element: column is fixed; rows vary in the tile
        int b_row = tile_start + threadIdx.y;
        if (b_row < N)
            Bs[threadIdx.y][threadIdx.x] = __ldg(&B[b_row * N + col]);
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Determine the overlapping range of k indices contributing to C[row, col]
        int k_start = (tile_start > col) ? tile_start : col;
        int k_end = ((tile_start + BLOCK_SIZE) < (row + 1)) ? (tile_start + BLOCK_SIZE) : (row + 1);

        for (int k = k_start; k < k_end; ++k) {
            int k_tile = k - tile_start;
            sum += As[threadIdx.y][k_tile] * Bs[k_tile][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}

// Forward function with an additional parameter to choose block size.
// Supported block sizes (tile dimensions): 8, 16, 32. Other values default to 32.

at::Tensor forward(at::Tensor A, at::Tensor B, int block_size = 32) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have the same dimensions");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 block, grid;

    switch (block_size) {
    case 8:
        block = dim3(8, 8);
        grid = dim3((N + 8 - 1) / 8, (N + 8 - 1) / 8);
        triangular_mm_tiled_template_kernel<8><<<grid, block>>>(A.data_ptr<float>(),
                                                                B.data_ptr<float>(),
                                                                C.data_ptr<float>(),
                                                                N);
        break;
    case 16:
        block = dim3(16, 16);
        grid = dim3((N + 16 - 1) / 16, (N + 16 - 1) / 16);
        triangular_mm_tiled_template_kernel<16><<<grid, block>>>(A.data_ptr<float>(),
                                                                 B.data_ptr<float>(),
                                                                 C.data_ptr<float>(),
                                                                 N);
        break;
    case 32:
    default:
        block = dim3(32, 32);
        grid = dim3((N + 32 - 1) / 32, (N + 32 - 1) / 32);
        triangular_mm_tiled_template_kernel<32><<<grid, block>>>(A.data_ptr<float>(),
                                                                 B.data_ptr<float>(),
                                                                 C.data_ptr<float>(),
                                                                 N);
        break;
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Templated tiled lower triangular matrix multiplication with block size experiment (CUDA)",
          py::arg("A"), py::arg("B"), py::arg("block_size") = 32);
}
