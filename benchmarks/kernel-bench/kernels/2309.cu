#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define TILE_SIZE 64  // Increased tile size for better occupancy
#define WARPS_PER_TILE 4

__global__ void hybrid_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Thread identification within warp and block
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    
    // Global position
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Each warp handles multiple elements within the tile
    float thread_results[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int k_offset = t * TILE_SIZE;
        
        // Cooperative loading using __ldg
        for (int i = tid; i < TILE_SIZE * TILE_SIZE; i += blockDim.x * blockDim.y) {
            int tile_row = i / TILE_SIZE;
            int tile_col = i % TILE_SIZE;
            
            if (blockIdx.y * TILE_SIZE + tile_row < M && k_offset + tile_col < K) {
                As[tile_row][tile_col] = __ldg(&A[(blockIdx.y * TILE_SIZE + tile_row) * K + k_offset + tile_col]);
            } else {
                As[tile_row][tile_col] = 0.0f;
            }
            
            if (blockIdx.x * TILE_SIZE + tile_row < N && k_offset + tile_col < K) {
                Bs[tile_col][tile_row] = __ldg(&B[(blockIdx.x * TILE_SIZE + tile_row) * K + k_offset + tile_col]);
            } else {
                Bs[tile_col][tile_row] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute multiple elements per thread using warp-level parallelism
        #pragma unroll 4
        for (int w = 0; w < 4; w++) {
            int local_row = threadIdx.y + w * (TILE_SIZE/4);
            if (local_row < TILE_SIZE) {
                for (int k = 0; k < TILE_SIZE; k += 4) {
                    float4 a_vec = *reinterpret_cast<const float4*>(&As[local_row][k]);
                    float4 b_vec = *reinterpret_cast<const float4*>(&Bs[k][threadIdx.x]);
                    thread_results[w] += a_vec.x * b_vec.x + a_vec.y * b_vec.y + 
                                       a_vec.z * b_vec.z + a_vec.w * b_vec.w;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results
    #pragma unroll 4
    for (int w = 0; w < 4; w++) {
        int out_row = row + w * (TILE_SIZE/4);
        if (out_row < M && col < N) {
            C[out_row * N + col] = thread_results[w];
        }
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
    
    dim3 block(TILE_SIZE/4, TILE_SIZE/4);  // Adjusted for better occupancy
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    hybrid_matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid tiled matrix multiplication with warp-level optimization (CUDA)");
}