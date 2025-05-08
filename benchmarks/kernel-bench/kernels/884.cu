#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 16
#define WARP_SIZE 32
#define TILE_DIM (BLOCK_DIM * 2)

__global__ void matmul_kernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             const int M, const int K, const int N) {
    // Shared memory declaration with padding to avoid bank conflicts
    __shared__ float As[TILE_DIM][TILE_DIM + 1];  // +1 padding to avoid bank conflicts
    __shared__ float Bs[TILE_DIM][TILE_DIM + 1];  // +1 padding to avoid bank conflicts

    // Calculate global indices
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int laneId = tid % WARP_SIZE;
    
    // Calculate base indices for block
    const int blockStartM = blockIdx.y * TILE_DIM;
    const int blockStartN = blockIdx.x * TILE_DIM;

    // Initialize registers for accumulation
    float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    
    // Loop over K dimension tiles
    for (int tileK = 0; tileK < K; tileK += TILE_DIM) {
        // Load A tile - coalesced access pattern
        #pragma unroll
        for (int i = 0; i < TILE_DIM; i += BLOCK_DIM) {
            const int globalRow = blockStartM + threadIdx.y + i;
            const int globalCol = tileK + threadIdx.x;
            
            if (globalRow < M && globalCol < K) {
                As[threadIdx.y + i][threadIdx.x] = A[globalRow * K + globalCol];
            } else {
                As[threadIdx.y + i][threadIdx.x] = 0.0f;
            }
        }
        
        // Load B tile - coalesced access pattern
        #pragma unroll
        for (int i = 0; i < TILE_DIM; i += BLOCK_DIM) {
            const int globalRow = tileK + threadIdx.y + i;
            const int globalCol = blockStartN + threadIdx.x;
            
            if (globalRow < K && globalCol < N) {
                Bs[threadIdx.y + i][threadIdx.x] = B[globalRow * N + globalCol];
            } else {
                Bs[threadIdx.y + i][threadIdx.x] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute 2x2 output elements
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            const float aReg[2] = {
                As[threadIdx.y * 2][k],
                As[threadIdx.y * 2 + 1][k]
            };
            
            const float bReg[2] = {
                Bs[k][threadIdx.x * 2],
                Bs[k][threadIdx.x * 2 + 1]
            };
            
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    acc[i][j] += aReg[i] * bReg[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results back to global memory in a coalesced pattern
    const int outRow = blockStartM + threadIdx.y * 2;
    const int outCol = blockStartN + threadIdx.x * 2;
    
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        if (outRow + i < M) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                if (outCol + j < N) {
                    C[(outRow + i) * N + (outCol + j)] = acc[i][j];
                }
            }
        }
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    
    matmul_kernel<<<grid, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication (CUDA) with coalesced memory access");
}