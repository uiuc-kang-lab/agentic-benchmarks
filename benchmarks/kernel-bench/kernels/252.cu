#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define WARP_SIZE 32

__global__ void bmm_hybrid_optimized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int batch_size,
    const int M,
    const int K,
    const int N
) {
    const int b = blockIdx.z;
    const float* batch_A = A + b * M * K;
    const float* batch_B = B + b * K * N;
    float* batch_C = C + b * M * N;

    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    float sum = 0.0f;
    
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    #pragma unroll 4
    for (int t = 0; t < numTiles; t++) {
        const int tiledCol = t * TILE_SIZE + threadIdx.x;
        const int tiledRow = t * TILE_SIZE + threadIdx.y;

        if (row < M && tiledCol < K) {
            As[threadIdx.y][threadIdx.x] = batch_A[row * K + tiledCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (tiledRow < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = batch_B[tiledRow * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();

        #pragma unroll 8
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = __fmaf_rn(As[threadIdx.y][k], Bs[k][threadIdx.x], sum);
        }

        if (t < numTiles - 1) {
            __syncthreads();
        }
    }
    
    if (row < M && col < N) {
        batch_C[row * N + col] = sum;
    }
}

torch::Tensor forward_bmm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input tensors must be CUDA tensors");
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3, "Input tensors must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions must match");

    const int batch_size = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int N = B.size(2);

    auto C = torch::zeros({batch_size, M, N}, 
                         torch::TensorOptions()
                         .dtype(A.dtype())
                         .device(A.device()));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, 
              (M + TILE_SIZE - 1) / TILE_SIZE, 
              batch_size);

    bmm_hybrid_optimized_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}