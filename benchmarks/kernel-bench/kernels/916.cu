#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 128
#define TILE_SIZE 32

__global__ void matmul_warp_kernel(const float* A, const float* B, float* C, 
                                  const int M, const int N, const int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const unsigned int laneId = threadIdx.x % WARP_SIZE;
    
    const int row = blockIdx.y * TILE_SIZE + (threadIdx.x / TILE_SIZE);
    const int col = blockIdx.x * TILE_SIZE + (threadIdx.x % TILE_SIZE);
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Collaborative loading using all threads in the block
        if (threadIdx.x < TILE_SIZE) {
            for (int i = threadIdx.x; i < TILE_SIZE; i += TILE_SIZE) {
                if (row < M && (t * TILE_SIZE + i) < K)
                    As[threadIdx.x][i] = A[row * K + t * TILE_SIZE + i];
                else
                    As[threadIdx.x][i] = 0.0f;
                    
                if ((t * TILE_SIZE + threadIdx.x) < K && (blockIdx.x * TILE_SIZE + i) < N)
                    Bs[threadIdx.x][i] = B[(t * TILE_SIZE + threadIdx.x) * N + blockIdx.x * TILE_SIZE + i];
                else
                    Bs[threadIdx.x][i] = 0.0f;
            }
        }
        
        __syncthreads();
        
        if (row < M && col < N) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; k++) {
                sum += As[warpId][k] * Bs[k][laneId];
            }
        }
        
        __syncthreads();
    }
    
    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Write result
    if (row < M && col < N && laneId == 0) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    const int threadsPerBlock = BLOCK_SIZE;
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_warp_kernel<<<grid, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication with warp optimizations (CUDA)");
}