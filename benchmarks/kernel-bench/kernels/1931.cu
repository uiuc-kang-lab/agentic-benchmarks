#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define WARP_SIZE 32

__global__ void warp_aligned_tril_mm_kernel(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int global_row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int global_col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Pre-compute block-level conditions to reduce divergent branching
    const bool valid_block = (blockIdx.y >= blockIdx.x);
    float sum = 0.0f;
    
    if (valid_block) {
        const int start_tile = global_col / TILE_SIZE;
        const int end_tile = min(global_row / TILE_SIZE + 1, (N + TILE_SIZE - 1) / TILE_SIZE);
        
        for (int tile = start_tile; tile < end_tile; tile++) {
            // Collaborative loading of tiles using warp-aligned access
            const int tile_row = global_row;
            const int tile_col = tile * TILE_SIZE + lane_id;
            
            // Load entire rows/columns using warp-aligned operations
            if (tile_row < N && tile_col < N) {
                As[threadIdx.y][threadIdx.x] = A[tile_row * N + tile_col];
                Bs[threadIdx.y][threadIdx.x] = B[tile_col * N + global_col];
            } else {
                As[threadIdx.y][threadIdx.x] = 0.0f;
                Bs[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            __syncthreads();
            
            // Compute partial results using warp-aligned operations
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; k++) {
                const bool valid_k = (tile * TILE_SIZE + k) >= global_col && 
                                   (tile * TILE_SIZE + k) <= global_row;
                sum += valid_k ? As[threadIdx.y][k] * Bs[k][threadIdx.x] : 0.0f;
            }
            
            __syncthreads();
        }
    }
    
    // Write results with minimal branching
    if (global_row < N && global_col < N) {
        const bool is_lower = global_row >= global_col;
        C[global_row * N + global_col] = is_lower ? sum : 0.0f;
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

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE/WARP_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    warp_aligned_tril_mm_kernel<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Warp-aligned triangular matrix multiplication (CUDA)");
}