#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile size for shared memory tiling
#define TILE 32
#define PAD 2  // Padding to reduce bank conflicts
#define TILE_K 8  // Tile size for K dimension for better register usage

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
    
    // Shared memory for tile of A and tile of B with padding to reduce bank conflicts
    __shared__ float As[TILE][TILE + PAD];
    __shared__ float Bs[TILE][TILE + PAD];
    
    // Register array for accumulating partial results
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Loop over tiles along the K dimension
    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; t++) {
        // Prefetch next tile indices
        int next_a_col = (t + 1) * TILE_K + threadIdx.x;
        int next_b_row = (t + 1) * TILE_K + threadIdx.y;
        
        // Load current tile
        #pragma unroll 4
        for (int k = 0; k < TILE_K; k += 4) {
            int a_col = t * TILE_K + k + threadIdx.x;
            if (row < M && a_col < K) {
                float4 a_vec = *reinterpret_cast<const float4*>(&A[b * M * K + row * K + a_col]);
                As[threadIdx.y][threadIdx.x + k] = a_vec.x;
                if (k + 1 < TILE_K) As[threadIdx.y][threadIdx.x + k + 1] = a_vec.y;
                if (k + 2 < TILE_K) As[threadIdx.y][threadIdx.x + k + 2] = a_vec.z;
                if (k + 3 < TILE_K) As[threadIdx.y][threadIdx.x + k + 3] = a_vec.w;
            } else {
                As[threadIdx.y][threadIdx.x + k] = 0.0f;
                if (k + 1 < TILE_K) As[threadIdx.y][threadIdx.x + k + 1] = 0.0f;
                if (k + 2 < TILE_K) As[threadIdx.y][threadIdx.x + k + 2] = 0.0f;
                if (k + 3 < TILE_K) As[threadIdx.y][threadIdx.x + k + 3] = 0.0f;
            }
            
            int b_row = t * TILE_K + k + threadIdx.y;
            if (b_row < K && col < N) {
                float4 b_vec = *reinterpret_cast<const float4*>(&B[b * K * N + b_row * N + col]);
                Bs[threadIdx.y + k][threadIdx.x] = b_vec.x;
                if (k + 1 < TILE_K) Bs[threadIdx.y + k + 1][threadIdx.x] = b_vec.y;
                if (k + 2 < TILE_K) Bs[threadIdx.y + k + 2][threadIdx.x] = b_vec.z;
                if (k + 3 < TILE_K) Bs[threadIdx.y + k + 3][threadIdx.x] = b_vec.w;
            } else {
                Bs[threadIdx.y + k][threadIdx.x] = 0.0f;
                if (k + 1 < TILE_K) Bs[threadIdx.y + k + 1][threadIdx.x] = 0.0f;
                if (k + 2 < TILE_K) Bs[threadIdx.y + k + 2][threadIdx.x] = 0.0f;
                if (k + 3 < TILE_K) Bs[threadIdx.y + k + 3][threadIdx.x] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute the partial products with aggressive loop unrolling
        #pragma unroll
        for (int i = 0; i < TILE_K; i++) {
            float a_val = As[threadIdx.y][i];
            float b_val = Bs[i][threadIdx.x];
            sum[i & 3] += a_val * b_val;
        }
        
        __syncthreads();
    }
    
    // Combine partial results
    float final_sum = sum[0] + sum[1] + sum[2] + sum[3];
    
    // Write result back to global memory using vectorized store
    if (row < M && col < N) {
        C[b * M * N + row * N + col] = final_sum;
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
