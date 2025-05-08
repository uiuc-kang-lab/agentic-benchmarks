#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_SIZE 16
#define UNROLL_FACTOR 4

__global__ void warp_optimized_upper_matmul_kernel(const float* __restrict__ A,
                                                  const float* __restrict__ B,
                                                  float* __restrict__ C,
                                                  int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Diagonal block mapping to ensure uniform warp execution
    int block_offset = blockIdx.x * TILE_SIZE;
    int row = block_offset + threadIdx.y;
    int col_start = block_offset + threadIdx.x;
    
    // Process multiple tiles in diagonal band
    for (int tile = 0; tile < gridDim.x; ++tile) {
        int col = col_start + tile * TILE_SIZE;
        float sum = 0.0f;
        
        if (row < N && col < N && row <= col) {
            const int start_k = row;
            const int end_k = col;
            
            // Process tiles of size TILE_SIZE
            for (int t = start_k; t < end_k; t += TILE_SIZE) {
                // Collaborative loading of tiles into shared memory
                if (threadIdx.y < TILE_SIZE && threadIdx.x < TILE_SIZE) {
                    int shared_row = t + threadIdx.y;
                    int shared_col = t + threadIdx.x;
                    if (shared_row < N && shared_col < N) {
                        As[threadIdx.y][threadIdx.x] = A[row*N + shared_col];
                        Bs[threadIdx.y][threadIdx.x] = B[shared_row*N + col];
                    } else {
                        As[threadIdx.y][threadIdx.x] = 0.0f;
                        Bs[threadIdx.y][threadIdx.x] = 0.0f;
                    }
                }
                __syncthreads();
                
                // Compute partial results using shared memory
                #pragma unroll 8
                for (int k = 0; k < TILE_SIZE; ++k) {
                    if ((t + k) <= end_k) {
                        sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                    }
                }
                __syncthreads();
            }
            
            // Handle remaining elements
            for (int k = start_k + ((end_k - start_k) / TILE_SIZE) * TILE_SIZE; 
                 k <= end_k; ++k) {
                sum += __ldg(&A[row*N + k]) * __ldg(&B[k*N + col]);
            }
            
            C[row*N + col] = sum;
        }
    }
}

torch::Tensor warp_optimized_upper_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    int num_blocks = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    warp_optimized_upper_matmul_kernel<<<num_blocks, threads>>>(A.data_ptr<float>(),
                                                                B.data_ptr<float>(),
                                                                C.data_ptr<float>(),
                                                                N);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &warp_optimized_upper_matmul, "Warp-optimized upper triangular matmul");
}