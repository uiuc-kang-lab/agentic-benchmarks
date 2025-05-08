#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_SIZE 32

__global__ void coalesced_upper_triangular_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by + ty;
    int col = bx + tx;
    
    float sum = 0.0f;
    
    for (int k = 0; k < N; k += TILE_SIZE) {
        if (row < N && (k + tx) < N)
            As[ty][tx] = A[row * N + k + tx];
        else
            As[ty][tx] = 0.0f;
        
        if ((k + ty) < N && col < N)
            Bs[ty][tx] = B[(k + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
        
        __syncthreads();
        
        for (int i = 0; i < TILE_SIZE; ++i) {
            int global_k = k + i;
            if (global_k >= row && global_k <= col)
                sum += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }
    
    if (row < N && col < N && row <= col)
        C[row * N + col] = sum;
}

torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    coalesced_upper_triangular_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Coalesced upper triangular matrix multiplication");
}
