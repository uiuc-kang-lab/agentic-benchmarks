#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define TILE_SIZE 32  // Aligned with warp size

__global__ void matmul_kernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             const int N,
                             const int N_aligned) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_SIZE + ty;
    const int col = blockIdx.x * TILE_SIZE + tx;
    
    // Pre-compute array indices
    const int row_N = row * N;
    float value = 0.0f;

    // Compute aligned dimensions for uniform execution
    const int num_tiles = (N_aligned + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; ++t) {
        // Compute tile offset once
        const int tile_offset = t * TILE_SIZE;
        
        // Load data into shared memory using cooperative groups
        const int a_idx = row_N + tile_offset + tx;
        const int b_idx = (tile_offset + ty) * N + col;
        
        // Use cooperative groups for better synchronization
        auto block = cooperative_groups::this_thread_block();
        auto warp = cooperative_groups::tiled_partition<WARP_SIZE>(block);
        
        // Stage 1: Load first half of the tile
        if (ty < TILE_SIZE/2) {
            s_A[ty][tx] = (row < N && (tile_offset + tx) < N) ? A[a_idx] : 0.0f;
            s_B[ty][tx] = ((tile_offset + ty) < N && col < N) ? B[b_idx] : 0.0f;
        }
        
        // Start computing on first half while loading second half
        if (ty >= TILE_SIZE/2) {
            s_A[ty][tx] = (row < N && (tile_offset + tx) < N) ? A[a_idx] : 0.0f;
            s_B[ty][tx] = ((tile_offset + ty) < N && col < N) ? B[b_idx] : 0.0f;
        }
        
        block.sync();
        
        // Compute tile product with improved memory access pattern
        #pragma unroll 4
        for (int k = 0; k < TILE_SIZE; k += 4) {
            value = fmaf(s_A[ty][k], s_B[k][tx], value);
            value = fmaf(s_A[ty][k+1], s_B[k+1][tx], value);
            value = fmaf(s_A[ty][k+2], s_B[k+2][tx], value);
            value = fmaf(s_A[ty][k+3], s_B[k+3][tx], value);
        }
        
        __syncthreads();
    }

    // Write result using predication instead of branching
    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Inputs must be square matrices");
    TORCH_CHECK(A.size(0) == B.size(0), "Input dimensions must match");

    const int N = A.size(0);
    // Align dimensions to warp size for uniform execution
    const int N_aligned = ((N + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    auto C = torch::zeros({N, N}, A.options());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N_aligned + TILE_SIZE - 1) / TILE_SIZE,
                (N_aligned + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        N_aligned
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-aligned Matrix Multiplication (CUDA)");
}