#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define constants
#define WARP_SIZE 32
#define TILE_SIZE 32    // Tile size for partitioning the k-dimension
#define TILE_M 8        // Number of output rows computed per block tile
#define TILE_N 4        // Number of output columns computed per block tile

// Hybrid kernel: combines warp-level shuffle reduction (from kernel 2) with tiling (from kernel 1).
// Each warp computes one output element C[i, j] for the lower triangular multiplication:
//   C[i,j] = sum_{k=j}^{i} A[i,k] * B[k,j]  for i>=j, else 0.
// The summation over k is partitioned into tiles of size TILE_SIZE to improve memory locality.
// Within each tile, the warp’s lanes cooperatively load and sum partial contributions via __shfl_down_sync.

__global__ void hybrid_tiled_warp_triangular_mm_kernel(const float* __restrict__ A,
                                                          const float* __restrict__ B,
                                                          float* __restrict__ C,
                                                          int N) {
    // Each thread block is organized with blockDim.x = WARP_SIZE and blockDim.y = (TILE_M * TILE_N).
    // Each warp (identified by threadIdx.y) computes one output element in a block tile.
    int lane = threadIdx.x;            // Lane id within the warp [0, WARP_SIZE-1]
    int warp_id = threadIdx.y;           // Warp id within the block

    // Map warp_id to tile coordinates
    int warp_row = warp_id / TILE_N;     // row offset within the block tile
    int warp_col = warp_id % TILE_N;     // column offset within the block tile

    // Global row and column indices for C
    int i = blockIdx.y * TILE_M + warp_row;
    int j = blockIdx.x * TILE_N + warp_col;

    // Return if out-of-bound
    if (i >= N || j >= N) return;

    // For upper-triangular area, set result to 0
    if (i < j) {
        if (lane == 0) {
            C[i * N + j] = 0.f;
        }
        return;
    }

    // Compute the dot product: C[i, j] = sum_{k=j}^{i} A[i,k] * B[k,j]
    // We partition the k-dimension into tiles of size TILE_SIZE.
    float sum_total = 0.f;
    int k_start = j;
    int k_end = i;

    for (int tile_start = k_start; tile_start <= k_end; tile_start += TILE_SIZE) {
        int tile_end = min(tile_start + TILE_SIZE - 1, k_end);
        int tile_count = tile_end - tile_start + 1;

        // Each warp lane will process a subset of the tile in strides of WARP_SIZE.
        float tile_sum = 0.f;
        for (int offset = lane; offset < tile_count; offset += WARP_SIZE) {
            int k = tile_start + offset;
            float a_val = A[i * N + k];
            float b_val = B[k * N + j];
            tile_sum += a_val * b_val;
        }

        // Warp-level reduction within the warp to sum tile_sum from all lanes
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            tile_sum += __shfl_down_sync(0xffffffff, tile_sum, offset);
        }
        
        // Lane 0 accumulates the reduced sum for this tile
        if (lane == 0) {
            sum_total += tile_sum;
        }
    }

    // Lane 0 writes the final result
    if (lane == 0) {
        C[i * N + j] = sum_total;
    }
}

// C++ interface exposed to PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Configure grid and block dimensions
    // Each block computes a tile of size TILE_M x TILE_N output elements, each computed by one warp
    dim3 block(WARP_SIZE, TILE_M * TILE_N);
    int grid_x = (N + TILE_N - 1) / TILE_N;
    int grid_y = (N + TILE_M - 1) / TILE_M;
    dim3 grid(grid_x, grid_y);

    hybrid_tiled_warp_triangular_mm_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid tiled warp-level triangular matrix multiplication");
}
