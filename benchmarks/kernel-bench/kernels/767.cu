#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_SIZE 16

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by + ty;
    int col = bx + tx;
    
    float sum = 0.0f;
    
    for (int i = 0; i < K; i += TILE_SIZE) {
        if (row < M && (i + tx) < K)
            As[ty][tx] = A[row * K + i + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (col < N && (i + ty) < K)
            Bs[ty][tx] = B[(i + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        #pragma unroll 4
        for (int k = 0; k < TILE_SIZE; k++)
            sum = __fmaf_rn(As[ty][k], Bs[k][tx], sum);
            
        __syncthreads();
    }
    
    if (row < M && col < N)
        C[row * N + col] = sum;
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
    
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication (CUDA)");
}