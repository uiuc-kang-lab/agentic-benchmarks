#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_SIZE 32

__global__ void coalesced_memory_access_upper_triangular_kernel(const float* __restrict__ A,
                                                           const float* __restrict__ B,
                                                           float* __restrict__ C,
                                                           const int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1]; // Avoid bank conflicts by padding
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    if (row < N && col < N && row <= col) {
        for (int t = row / TILE_SIZE * TILE_SIZE; t <= col; t += TILE_SIZE) {
            if ((row < N) && (t + threadIdx.x < N) && (row <= t + threadIdx.x)) {
                As[threadIdx.y][threadIdx.x] = A[row * N + t + threadIdx.x];
            } else {
                As[threadIdx.y][threadIdx.x] = 0.0f;
            }

            if ((t + threadIdx.y < N) && (col < N)) { 
                Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
            } else {
                Bs[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            __syncthreads();

            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                int global_k = t + k;
                if (global_k >= row && global_k <= col && global_k < N) {
                    sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                }
            }

            __syncthreads();
        }

        if (row < N && col < N && row <= col) {
            C[row * N + col] = sum;
        }
    }
}

torch::Tensor coalesced_memory_access_upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);
    
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE,
                   (N + TILE_SIZE - 1) / TILE_SIZE);
    
    coalesced_memory_access_upper_triangular_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &coalesced_memory_access_upper_triangular_matmul, "Coalesced memory access upper triangular matrix multiplication");
}
