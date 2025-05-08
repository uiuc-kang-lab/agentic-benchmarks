#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define WARP_SIZE 32

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Compute mask for valid elements (1.0 for valid, 0.0 for invalid)
    const float mask = (row >= col && row < N && col < N) ? 1.0f : 0.0f;
    float sum = 0.0f;
    
    // Calculate the number of tiles needed
    const int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    // Pre-calculate the starting tile based on column position
    const int start_tile = col / TILE_SIZE;
    
    // Process tiles
    for (int t = start_tile; t < num_tiles; t++) {
        // Load tile data into shared memory
        const int global_x = t * TILE_SIZE + tx;
        const int global_y = t * TILE_SIZE + ty;
        
        // Use vectorized loads when possible
        if (row < N && global_x < N) {
            As[ty][tx] = A[row * N + global_x];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (global_y < N && col < N) {
            Bs[ty][tx] = B[global_y * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial results
        #pragma unroll 8
        for (int k = 0; k < TILE_SIZE; k++) {
            // Use mask to handle boundary conditions without branching
            const float valid_k = (t * TILE_SIZE + k <= row) ? 1.0f : 0.0f;
            sum += As[ty][k] * Bs[k][tx] * valid_k;
        }
        
        __syncthreads();
    }
    
    // Write result using mask to avoid branching
    if (row < N && col < N) {
        C[row * N + col] = sum * mask;
    }
}

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
    
    // Ensure block dimensions are aligned with warp size
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                   (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Divergence-free triangular matrix multiplication (CUDA)");
}