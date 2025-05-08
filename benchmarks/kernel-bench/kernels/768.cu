#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define WARP_SIZE 32
#define TILE_SIZE 16

__global__ void matmul_warp_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const unsigned int FULL_MASK = 0xffffffff;
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Compute matrix multiplication
    for (int i = 0; i < K; i += TILE_SIZE) {
        // Load data into shared memory
        if (row < M && (i + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + i + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if ((i + threadIdx.y) < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(i + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = __fmaf_rn(As[threadIdx.y][k], Bs[k][threadIdx.x], sum);
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    torch::Tensor C = torch::zeros({M, N}, A.options());
    
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE,
                   (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_warp_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-optimized Matrix Multiplication (CUDA)");
}