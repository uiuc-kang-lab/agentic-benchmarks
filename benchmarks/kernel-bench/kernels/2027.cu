#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Define tile size – you can adjust this for your hardware
#define TILE_SIZE 32

// This kernel combines shared-memory tiling (from Kernel 1) with an even workload mapping (from Kernel 2) 
// by launching only lower-triangular blocks using a 1D grid. Each block corresponds to a tile in the lower
// triangular region of C. In diagonal blocks, threads with row < col are discarded, ensuring correct results.

__global__ void triangular_mm_tiled_even_kernel(const float* __restrict__ A,
                                                 const float* __restrict__ B,
                                                 float* __restrict__ C,
                                                 int N) {
    // Map the 1D block index to lower-triangular tile indices (tile_i, tile_j) such that tile_i >= tile_j.
    int blockId = blockIdx.x;  // Linear block id
    // Using triangular number inversion: blockId = tile_i*(tile_i+1)/2 + tile_j
    float tmp = sqrtf(8.0f * (float)blockId + 1.0f);
    int tile_i = (int)((tmp - 1.0f) * 0.5f);
    int tile_j = blockId - tile_i * (tile_i + 1) / 2;

    // Compute global row and column for this thread
    int row = tile_i * TILE_SIZE + threadIdx.y;
    int col = tile_j * TILE_SIZE + threadIdx.x;

    // Check bounds
    if (row >= N || col >= N)
        return;

    // For lower-triangular matrices, we only compute elements where row >= col.
    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Shared memory for a tile, with extra column to avoid bank conflicts
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    // Loop over tiles in the k dimension; the dot product runs from k = col to k = row (inclusive)
    for (int t = 0; t < numTiles; t++) {
        int tile_start = t * TILE_SIZE;
        // Since k runs only up to row, we can break early if tile_start exceeds row
        if (tile_start > row)
            break;

        // Load one element for A: row remains the same, and columns from tile_start are loaded
        int a_col = tile_start + threadIdx.x;
        if (a_col < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load one element for B: column remains the same, and rows from tile_start are loaded
        int b_row = tile_start + threadIdx.y;
        if (b_row < N)
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Determine the sub-range within this tile that contributes to the dot product for C[row, col]
        // We need k in [max(tile_start, col), min(tile_start + TILE_SIZE, row + 1)).
        int k_start = (tile_start < col) ? col : tile_start;
        int k_end = ((tile_start + TILE_SIZE) < (row + 1)) ? (tile_start + TILE_SIZE) : (row + 1);

        for (int k = k_start; k < k_end; ++k) {
            sum += As[threadIdx.y][k - tile_start] * Bs[k - tile_start][threadIdx.x];
        }

        __syncthreads();
    }

    C[row * N + col] = sum;
}

// PyTorch interface
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have the same dimensions");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Determine the number of tiles along one dimension
    int numTile = (N + TILE_SIZE - 1) / TILE_SIZE;
    // Total number of lower-triangular blocks: numTile*(numTile+1)/2
    int totalBlocks = numTile * (numTile + 1) / 2;

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(totalBlocks);

    triangular_mm_tiled_even_kernel<<<grid, block>>>(A.data_ptr<float>(),
                                                     B.data_ptr<float>(),
                                                     C.data_ptr<float>(),
                                                     N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tiled and even-workload lower triangular matrix multiplication (CUDA)");
}
