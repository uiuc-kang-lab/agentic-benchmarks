#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

// Tiled kernel for lower-triangular matrix multiplication with shared memory
// Computes C = tril(A * B) where A and B are lower-triangular.
// Each thread computes an element C[row,col] (only valid if row >= col).

__global__ void tiled_triangular_mm_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int N) {
    // Compute global row and column indices
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Check boundaries and ensure we only compute the lower triangular part
    if (row >= N || col >= N) return;
    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;

    // Shared memory tiles for A and B
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Loop over tiles in the k dimension. For C[row,col], valid k indices are from col to row.
    // We process these indices in chunks of BLOCK_SIZE.
    for (int k_tile = col; k_tile <= row; k_tile += BLOCK_SIZE) {
        // Each thread loads one element of A and one element of B into shared memory
        int a_index = k_tile + threadIdx.x;  // Column index for A
        if (a_index <= row && a_index < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + a_index];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int b_index = k_tile + threadIdx.y;  // Row index for B
        // For B, valid if b_index is within bounds and due to lower-triangularity, b_index >= col
        if (b_index < N && b_index <= row) {
            Bs[threadIdx.y][threadIdx.x] = B[b_index * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize to ensure the tile is loaded
        __syncthreads();

        // Compute how many elements in this tile are valid
        int tile_width = BLOCK_SIZE;
        if (k_tile + BLOCK_SIZE - 1 > row) {
            tile_width = row - k_tile + 1;
        }

        // Accumulate product over the valid range in the tile
        #pragma unroll
        for (int k = 0; k < tile_width; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Only synchronize if there will be another iteration to load a new tile
        if (k_tile + BLOCK_SIZE <= row) {
            __syncthreads();
        }
    }

    C[row * N + col] = sum;
}

// C++ interface exposed to PyTorch using pybind11
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    int grid_dim = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blocks(grid_dim, grid_dim);

    tiled_triangular_mm_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Tiled Triangular Matrix Multiplication with Shared Memory and Minimal Synchronization (CUDA)");
}
