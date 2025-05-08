#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// This kernel computes C = tril(A * B) for lower triangular matrices A and B
// using shared memory tiling. Global memory accesses are aligned so that threads in a warp
// read consecutive memory locations from A and B. The kernel loads tiles of A and B into
// shared memory for reuse and then computes partial sums, applying a condition to accumulate
// only for indices within the valid triangular range (k between j and i).

__global__ void aligned_triangular_mm_kernel(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               int N) {
    // Compute block indices
    int blockRow = blockIdx.y;  // for output row
    int blockCol = blockIdx.x;  // for output col
    
    // Process only lower triangular blocks
    if (blockRow < blockCol) return;
    
    // Global row and column indices for C
    int row = blockRow * TILE_SIZE + threadIdx.y; // i index
    int col = blockCol * TILE_SIZE + threadIdx.x;   // j index
    
    float acc = 0.0f;

    // Loop over tiles of the summation index k
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    // Shared memory buffers for tiles of A and B
    __shared__ float shA[TILE_SIZE][TILE_SIZE];
    __shared__ float shB[TILE_SIZE][TILE_SIZE];

    for (int t = 0; t < numTiles; t++) {
        // Load a tile from A: each thread loads one element
        int col_index = t * TILE_SIZE + threadIdx.x; // column index for A
        if (row < N && col_index < N)
            shA[threadIdx.y][threadIdx.x] = A[row * N + col_index];
        else
            shA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load a tile from B: each thread loads one element
        int row_index = t * TILE_SIZE + threadIdx.y; // row index for B
        if (row_index < N && col < N)
            shB[threadIdx.y][threadIdx.x] = B[row_index * N + col];
        else
            shB[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        // Accumulate partial results from the tile
        // Global k index corresponding to k_local in the tile
        for (int k_local = 0; k_local < TILE_SIZE; k_local++) {
            int global_k = t * TILE_SIZE + k_local;
            // Only include contribution if k is in the valid range for lower triangular multiplication
            // That is, for output C[i,j], accumulate only when global_k is between j and i
            if (global_k >= col && global_k <= row) {
                acc += shA[threadIdx.y][k_local] * shB[k_local][threadIdx.x];
            }
        }
        __syncthreads();
    }
    
    // Write the result if within valid range
    if (row < N && col < N) {
        C[row * N + col] = acc;
    }
}

// C++ interface exposed to PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same dimensions");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    int gridX = (N + TILE_SIZE - 1) / TILE_SIZE;
    int gridY = (N + TILE_SIZE - 1) / TILE_SIZE;
    dim3 gridDim(gridX, gridY);
    
    aligned_triangular_mm_kernel<<<gridDim, blockDim>>>(
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
    m.def("forward", &forward, "Aligned shared memory and coalesced accesses triangular matmul (CUDA)");
}
