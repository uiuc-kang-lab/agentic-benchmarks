#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// This CUDA kernel computes C = tril(A * B) for lower triangular matrices A and B using shared memory tiling.
// Each thread computes one element C[row, col] where row >= col. The kernel loads tiles of the relevant rows of A and
// columns of B into shared memory to reduce global memory latency. The summation is only performed over the valid
// range k in [col, row] (and within the current tile).
__global__ void shared_tiled_lower_triangular_kernel(const float* __restrict__ A,
                                                       const float* __restrict__ B,
                                                       float* __restrict__ C,
                                                       int N) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row >= N || col >= N)
        return;

    // Only compute lower triangular part; upper triangular elements are zero.
    if (row < col) {
        C[row * N + col] = 0.f;
        return;
    }

    float sum = 0.f;

    // Allocate shared memory for tiles of A and B.
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        int tile_start = t * TILE_SIZE;

        // Load tile of A: We load A[row, tile_start + threadIdx.x] if within bounds and if it lies within the lower-triangular region
        int a_col = tile_start + threadIdx.x;
        if (a_col < N && a_col <= row)
            As[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.f;

        // Load tile of B: We load B[tile_start + threadIdx.y, col] if within bounds and if it lies within the lower-triangular structure
        int b_row = tile_start + threadIdx.y;
        if (b_row < N && b_row >= col)
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.f;

        __syncthreads();

        // Determine the valid summation range in this tile for k, as k must satisfy col <= k <= row.
        int k_tile_start = (tile_start < col) ? col : tile_start;
        int k_tile_end = ((tile_start + TILE_SIZE) < (row + 1)) ? (tile_start + TILE_SIZE) : (row + 1);

        for (int k = k_tile_start; k < k_tile_end; ++k) {
            int idx = k - tile_start;  // index inside the tile
            sum += As[threadIdx.y][idx] * Bs[idx][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}

// PyTorch interface function: validates inputs, sets up launch configuration, and calls the kernel.
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

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    shared_tiled_lower_triangular_kernel<<<grid, block>>>(
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
    m.def("forward", &forward, "Shared memory tiled lower triangular matrix multiplication (CUDA)");
}
