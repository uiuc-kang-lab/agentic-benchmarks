#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile size for shared memory tiling
#define TILE 16

// CUDA kernel for batched matrix multiplication using 2D tiling and 3D grid indexing
// Computes C = A * B, where A: (batch_size, M, K), B: (batch_size, K, N), and C: (batch_size, M, N)
__global__ void bmm_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    // Determine the batch index from the z-dimension of the grid.
    int b = blockIdx.z;
    
    // Determine the row and column of the C matrix element to work on
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    
    // Shared memory for tile of A and tile of B
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    
    float sum = 0.0f;
    // Loop over tiles along the K dimension
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // Load element from A into shared memory if within bounds
        int a_col = t * TILE + threadIdx.x;
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[b * M * K + row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load element from B into shared memory if within bounds
        int b_row = t * TILE + threadIdx.y;
        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b * K * N + b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute the partial product for this tile
        for (int i = 0; i < TILE; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result back to global memory
    if (row < M && col < N) {
        C[b * M * N + row * N + col] = sum;
    }
}

// Forward function to launch the tiled batched matrix multiplication kernel
torch::Tensor forward_bmm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions (K) must match");

    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({batch_size, M, N}, options);

    // Define block dimensions and grid dimensions
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, batch_size);

    bmm_tiled_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Batched matrix multiplication with tiling (CUDA)");
}
