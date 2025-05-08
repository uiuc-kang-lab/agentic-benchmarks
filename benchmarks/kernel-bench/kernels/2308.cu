#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define TILE_SIZE 64  // Increased tile size for better occupancy
#define WARPS_PER_TILE 4

__global__ void matmul_hybrid_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Each thread block handles a TILE_SIZE x TILE_SIZE output tile
    // Within each tile, warps cooperate to compute partial results
    int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    int tile_row = blockIdx.y * TILE_SIZE;
    int tile_col = blockIdx.x * TILE_SIZE;
    
    // Each warp is responsible for a portion of the tile
    int warp_row = warp_id / WARPS_PER_TILE;
    int warp_col = warp_id % WARPS_PER_TILE;
    
    // Calculate global indices
    int row = tile_row + warp_row * (TILE_SIZE/WARPS_PER_TILE) + (lane_id / (TILE_SIZE/WARPS_PER_TILE));
    int col = tile_col + warp_col * (TILE_SIZE/WARPS_PER_TILE) + (lane_id % (TILE_SIZE/WARPS_PER_TILE));
    
    float c_val = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int k_offset = t * TILE_SIZE;
        
        // Cooperative loading of tiles using __ldg
        for (int i = threadIdx.x + threadIdx.y * blockDim.x; 
             i < TILE_SIZE * TILE_SIZE; 
             i += blockDim.x * blockDim.y) {
            int tile_i = i / TILE_SIZE;
            int tile_j = i % TILE_SIZE;
            
            if ((tile_row + tile_i) < M && (k_offset + tile_j) < K) {
                As[tile_i][tile_j] = __ldg(&A[(tile_row + tile_i) * K + k_offset + tile_j]);
            } else {
                As[tile_i][tile_j] = 0.0f;
            }
            
            if ((tile_col + tile_i) < N && (k_offset + tile_j) < K) {
                Bs[tile_j][tile_i] = __ldg(&B[(tile_col + tile_i) * K + k_offset + tile_j]);
            } else {
                Bs[tile_j][tile_i] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute partial results using warp-level parallelism
        #pragma unroll 8
        for (int k = 0; k < TILE_SIZE; ++k) {
            c_val += As[warp_row * (TILE_SIZE/WARPS_PER_TILE) + (lane_id / (TILE_SIZE/WARPS_PER_TILE))][k] * 
                     Bs[k][warp_col * (TILE_SIZE/WARPS_PER_TILE) + (lane_id % (TILE_SIZE/WARPS_PER_TILE))];
        }
        
        __syncthreads();
    }
    
    // Write results
    if (row < M && col < N) {
        C[row * N + col] = c_val;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "Inner dimensions must match");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());
    
    dim3 block(WARP_SIZE, WARPS_PER_TILE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_hybrid_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid warp-tile matrix multiplication (CUDA)");
}