#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_SIZE 16
#define UNROLL_FACTOR 4

__global__ void warp_optimized_upper_matmul_kernel(const float* __restrict__ A,
                                                  const float* __restrict__ B,
                                                  float* __restrict__ C,
                                                  int N) {
    // Diagonal block mapping to ensure uniform warp execution
    int block_offset = blockIdx.x * TILE_SIZE;
    int row = block_offset + threadIdx.y;
    int col_start = block_offset + threadIdx.x;
    
    // Process multiple tiles in diagonal band
    for (int tile = 0; tile < gridDim.x; ++tile) {
        int col = col_start + tile * TILE_SIZE;
        
        if (row < N && col < N && row <= col) {
            float sum = 0.0f;
            const int start_k = row;
            const int end_k = col;
            
            // Vectorized loads for aligned access
            const float* A_ptr = &A[row*N + start_k];
            const bool aligned = ((uintptr_t)A_ptr % 16) == 0;
            
            int k = start_k;
            if (aligned && (end_k - start_k + 1) >= UNROLL_FACTOR) {
                const float4* A_vec = reinterpret_cast<const float4*>(A_ptr);
                const int vec_steps = (end_k - start_k + 1) / UNROLL_FACTOR;
                
                #pragma unroll
                for (int i = 0; i < vec_steps; ++i) {
                    float4 a_chunk = __ldg(A_vec + i);
                    sum += a_chunk.x * __ldg(&B[(k + i*UNROLL_FACTOR)*N + col]);
                    sum += a_chunk.y * __ldg(&B[(k + i*UNROLL_FACTOR + 1)*N + col]);
                    sum += a_chunk.z * __ldg(&B[(k + i*UNROLL_FACTOR + 2)*N + col]);
                    sum += a_chunk.w * __ldg(&B[(k + i*UNROLL_FACTOR + 3)*N + col]);
                }
                k += vec_steps * UNROLL_FACTOR;
            }
            
            // Process remaining elements
            #pragma unroll
            for (; k <= end_k; ++k) {
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